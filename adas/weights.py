import os
import sys
import urllib.request

YOLOPV2_URL = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"

def ensure_yolopv2(path="weights/yolopv2.pt", url=YOLOPV2_URL):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    print(f"Chưa có {path}, đang tải YOLOPv2 (~150 MB)...")

    shown = [-1]

    def progress(blocks, block_size, total):
        if total > 0:
            mb = int(min(blocks * block_size, total) / 1e6)
            if mb // 10 != shown[0] // 10 or mb == int(total / 1e6):
                shown[0] = mb
                sys.stdout.write(f"\r  {mb}/{total / 1e6:.0f} MB")
                sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, progress)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SystemExit(f"\nKhông tải được YOLOPv2 ({e}).\nTải tay tại {url} rồi bỏ vào {path}.")
    os.replace(tmp, path)
    print("\n  Tải xong.")
    return path


def resolve_light_cls(value, default_path="weights/light_cls.pt"):
    if value == "auto":
        return default_path if os.path.isfile(default_path) else None
    return value or None
