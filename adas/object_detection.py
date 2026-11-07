import cv2
import numpy as np

COCO_KEEP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}
VEHICLES = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
}

def traffic_light_color(frame_bgr, box):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 2:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (sat >= 90) & (val >= 120)

    counts = {
        "red": int(np.count_nonzero(lit & ((hue <= 10) | (hue >= 160)))),
        "yellow": int(np.count_nonzero(lit & (hue >= 15) & (hue <= 35))),
        "green": int(np.count_nonzero(lit & (hue >= 40) & (hue <= 95))),
    }
    color, n = max(counts.items(), key=lambda kv: kv[1])
    if n < max(3, 0.03 * crop.shape[0] * crop.shape[1]):
        return "unknown"
    return color


class ObjectDetector:
    def __init__(self, weights="weights/yolov8s.pt", device="cuda:0", light_cls=None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.light_model = YOLO(light_cls) if light_cls else None

    def _classify_lights(self, frame_bgr, lights, pad=0.15):
        h, w = frame_bgr.shape[:2]
        crops, owners = [], []
        for d in lights:
            x1, y1, x2, y2 = d["box"]
            px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
            crop = frame_bgr[max(y1 - py, 0):min(y2 + py, h), max(x1 - px, 0):min(x2 + px, w)]
            if crop.size == 0:
                d["color"] = "unknown"
                continue
            crops.append(crop)
            owners.append(d)
        if not crops:
            return
        results = self.light_model.predict(crops, device=self.device, verbose=False)
        for d, r in zip(owners, results):
            name = r.names[int(r.probs.top1)]
            d["color"] = name if name in ("red", "yellow", "green") else "unknown"

    def __call__(self, frame_bgr, conf=0.35):
        res = self.model.predict(
            frame_bgr,
            conf=conf,
            classes=list(COCO_KEEP),
            device=self.device,
            verbose=False,
        )[0]
        dets = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()
        for box, c, s in zip(xyxy, cls, scores):
            det = {
                "box": tuple(int(v) for v in box),
                "name": COCO_KEEP[int(c)],
                "conf": float(s),
            }
            if det["name"] == "traffic light" and self.light_model is None:
                det["color"] = traffic_light_color(frame_bgr, det["box"])
            dets.append(det)
        if self.light_model is not None:
            self._classify_lights(frame_bgr, [d for d in dets if d["name"] == "traffic light"])
        return dets