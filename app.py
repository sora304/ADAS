import argparse
import os
import tempfile

import cv2
import gradio as gr
import imageio

TITLE = "Hệ thống hỗ trợ lái xe, cảnh báo vật cản trong làn và đèn tín hiệu giao thông"
SUBTITLE = "YOLOv8 + YOLOPv2"

PREVIEW_EVERY = 5
GRADIO6 = int(gr.__version__.split(".")[0]) >= 6
THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

LEVELS = {
    "danger": ("rgba(226,75,74,0.16)", "#E24B4A"),
    "warning": ("rgba(239,159,39,0.18)", "#BA7517"),
    "ok": ("rgba(99,153,34,0.16)", "#639922"),
    "info": ("rgba(55,138,221,0.14)", "#378ADD"),
    "idle": ("rgba(128,128,128,0.12)", "inherit"),
}
CODE_VI = {
    "RED LIGHT": ("danger", "đèn đỏ"),
    "YELLOW LIGHT": ("warning", "đèn vàng"),
    "PERSON IN LANE": ("danger", "người trong làn"),
    "VEHICLE IN LANE": ("warning", "xe trong làn"),
}
LIGHT_CARD = {"red": ("Đỏ", "#E24B4A"), "yellow": ("Vàng", "#BA7517"), "green": ("Xanh", "#639922")}


def _box(level, text, size=16, weight=500):
    bg, fg = LEVELS[level]
    return (f'<div style="background:{bg};color:{fg};border-radius:12px;padding:12px 16px;'
            f'font-size:{size}px;font-weight:{weight}">{text}</div>')


IDLE_HTML = _box("idle", "Chưa chạy · chọn ảnh hoặc video rồi bấm Chạy nhận diện", weight=400)


def _badge(text, level):
    bg, fg = LEVELS[level]
    return (f'<span style="background:{bg};color:{fg};border-radius:8px;padding:4px 10px;'
            f'font-size:13px;white-space:nowrap">{text}</span>')


def header_html(device, light_method):
    gpu = str(device).startswith("cuda")
    badges = (_badge(f"GPU · {device}" if gpu else "CPU · sẽ chậm", "ok" if gpu else "warning"))
    return ('<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;'
            'flex-wrap:wrap;padding:4px 0 8px">'
            f'<div><div style="font-size:22px;font-weight:600">{TITLE}</div>'
            f'<div style="opacity:.7;margin-top:2px">{SUBTITLE}</div></div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap">{badges}</div></div>')


def _swatch(css, label):
    return (f'<span style="display:inline-flex;align-items:center;gap:6px">'
            f'<span style="display:inline-block;{css}"></span>{label}</span>')


LEGEND_HTML = ('<div style="display:flex;gap:6px 16px;flex-wrap:wrap;font-size:13px;opacity:.85">'
               + _swatch("width:12px;height:12px;background:#97C459;border-radius:2px", "Làn đang chạy")
               + _swatch("width:12px;height:12px;background:#85B7EB;border-radius:2px", "Mặt đường khác")
               + _swatch("width:14px;height:4px;background:#EF9F27", "Vạch kẻ làn")
               + _swatch("width:10px;height:10px;border:2px solid #E24B4A", "Trong làn")
               + _swatch("width:10px;height:10px;border:2px solid #378ADD", "Người")
               + _swatch("width:10px;height:10px;border:2px solid #888780", "Xe")
               + "</div>")


def alert_html(alerts):
    if not alerts:
        return _box("ok", "An toàn · không có cảnh báo")
    level = "danger" if any(a[0] == "danger" for a in alerts) else "warning"
    return _box(level, " · ".join(a[1] for a in alerts))


def video_alert_html(counts):
    if not counts:
        return _box("ok", "Cả video không có cảnh báo")
    level = "danger" if any(CODE_VI[c][0] == "danger" for c in counts) else "warning"
    parts = [f"{CODE_VI[c][1]} ({n} frame)" for c, n in counts.items()]
    return _box(level, "Trong video có: " + " · ".join(parts))


def stats_html(n_veh=None, n_person=None, light=None, speed=None):
    light_text, light_color = LIGHT_CARD.get(light, ("—", "inherit"))
    cards = [("Xe", "—" if n_veh is None else n_veh, "inherit"),
             ("Người", "—" if n_person is None else n_person, "inherit"),
             ("Đèn", light_text, light_color),
             ("Tốc độ", speed or "—", "inherit")]
    html = '<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px">'
    for label, value, color in cards:
        html += ('<div style="background:rgba(128,128,128,0.12);border-radius:10px;padding:8px 12px">'
                 f'<div style="opacity:.7;font-size:13px">{label}</div>'
                 f'<div style="font-size:20px;font-weight:600;color:{color}">{value}</div></div>')
    return html + "</div>"

def build_ui(pipe):
    def to_rgb(bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def run(mode, img, vid, conf, progress=gr.Progress()):

        if mode == "image":
            if img is None:
                raise gr.Error("Chưa chọn ảnh.")
            vis, info = pipe.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), conf)
            yield (gr.update(value=to_rgb(vis), visible=True), gr.update(value=None, visible=False),
                   alert_html(info["alerts"]),
                   stats_html(info["n_vehicles"], info["n_persons"], info["light"], f"{info['ms']:.0f} ms"),
                   gr.update(value="", visible=False))
            return

        if not vid:
            raise gr.Error("Chưa chọn video.")
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        out_path = os.path.join(tempfile.mkdtemp(), "ket_qua.mp4")
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=2)
        pipe.reset()
        n = n_red = n_inlane = 0
        t_sum = 0.0
        counts = {}
        info = None
        progress(0, desc="Bắt đầu xử lý video")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                vis, info = pipe.process(frame, conf, smooth=True)
                writer.append_data(to_rgb(vis))
                n += 1
                t_sum += info["ms"]
                n_red += info["light"] == "red"
                n_inlane += bool(info["in_lane"])
                for a in info["alerts"]:
                    counts[a[2]] = counts.get(a[2], 0) + 1
                if n % PREVIEW_EVERY == 1:
                    progress(n / total if total else None, desc=f"Frame {n}/{total or '?'}")
                    yield (gr.update(value=to_rgb(vis), visible=True), gr.update(visible=False),
                           alert_html(info["alerts"]),
                           stats_html(info["n_vehicles"], info["n_persons"], info["light"],
                                      f"{1000 * n / t_sum:.0f} FPS"),
                           gr.update(visible=False))
        finally:
            writer.close()
            cap.release()

        if n == 0:
            raise gr.Error("Không đọc được frame nào từ video.")
        avg_fps = 1000 * n / t_sum
        summary = (f"Video {n} frame: đèn đỏ {100 * n_red / n:.0f}% · vật thể trong làn {100 * n_inlane / n:.0f}%"
                   f" · tốc độ trung bình {avg_fps:.0f} FPS. Bấm biểu tượng tải xuống trên video để lưu kết quả.")
        yield (gr.update(visible=False), gr.update(value=out_path, visible=True), video_alert_html(counts),
               stats_html(info["n_vehicles"], info["n_persons"], info["light"], f"{avg_fps:.0f} FPS"),
               gr.update(value=_box("info", summary, size=14, weight=400), visible=True))

    blocks_kwargs = {"title": TITLE} if GRADIO6 else {"title": TITLE, "theme": THEME}
    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(header_html(pipe.device, getattr(pipe, "light_method", "hsv")))
        mode = gr.State("image")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Ảnh") as tab_img:
                        in_img = gr.Image(type="numpy", label="Ảnh đầu vào", height=300)
                    with gr.Tab("Video") as tab_vid:
                        in_vid = gr.Video(label="Xem trước · hoặc kéo thả file khác", height=260)
                with gr.Group():
                    conf = gr.Slider(0.05, 0.9, value=0.35, step=0.05, label="Ngưỡng tin cậy",
                                     info="Nên để 0,3–0,4")
                btn = gr.Button("Chạy nhận diện", variant="primary", size="lg")
            with gr.Column(scale=3):
                alert = gr.HTML(IDLE_HTML)
                out_img = gr.Image(label="Kết quả", interactive=False, height=420)
                out_vid = gr.Video(label="Video kết quả", interactive=False, visible=False, height=420)
                gr.HTML(LEGEND_HTML)
                stats = gr.HTML(stats_html())
                summary = gr.HTML(visible=False)

        tab_img.select(lambda: "image", None, mode)
        tab_vid.select(lambda: "video", None, mode)
        btn.click(run, [mode, in_img, in_vid, conf], [out_img, out_vid, alert, stats, summary])
    return demo


def launch(demo, port=None):
    kwargs = {
        "server_name": "127.0.0.1",
    }

    if port is not None:
        kwargs["server_port"] = port

    if GRADIO6:
        kwargs["theme"] = THEME

    demo.launch(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="weights/yolov8s.pt", help="weights YOLOv8 (tự tải nếu chưa có)")
    ap.add_argument("--yolopv2", default="weights/yolopv2.pt", help="weights YOLOPv2")
    ap.add_argument("--device", default=None, help="vd: cuda:0 hoặc cpu (mặc định tự chọn)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--light-cls", default="auto", help="model màu đèn; auto = dùng weights/light_cls.pt nếu có, \"\" = HSV")
    args = ap.parse_args()

    from adas.pipeline import ADASPipeline
    from adas.weights import ensure_yolopv2, resolve_light_cls

    ensure_yolopv2(args.yolopv2)
    pipe = ADASPipeline(args.yolo, args.yolopv2, args.device, resolve_light_cls(args.light_cls))
    if pipe.device == "cpu":
        print("[!] Đang chạy CPU, sẽ chậm. Kiểm tra lại bản PyTorch có CUDA (xem README).")
    print("Đang warmup model...")
    pipe.warmup()
    launch(build_ui(pipe), args.port)

if __name__ == "__main__":
    main()
