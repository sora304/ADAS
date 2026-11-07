import os
import unicodedata
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEXT_COLOR = (0, 0, 0)

EGO_COLOR = (0, 255, 0)
OTHER_COLOR = (0, 255, 0)
LINE_COLOR = (0, 0, 255)

BOX_IN_LANE = (0, 0, 255)
BOX_PERSON = (128, 0, 128)
BOX_VEHICLE = (0, 255, 255)
LIGHT_COLORS = {"red": (74, 75, 226), "yellow": (39, 159, 239), "green": (34, 153, 99), "unknown": (128, 135, 136)}
BANNER_COLOR = (0, 0, 255)

NAMES_VI = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
}
LIGHTS_VI = {
    "red": "red light",
    "yellow": "yellow light",
    "green": "green light",
    "unknown": "unknown"}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]


@lru_cache(maxsize=None)
def _font(size):
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return None


def to_ascii(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _blend(img, mask, color, alpha):
    if mask is not None and mask.any():
        img[mask] = (img[mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)


def _rgb(bgr):
    return bgr[2], bgr[1], bgr[0]


def _draw_texts(img, labels, size):
    font = _font(size)
    if font is None:
        scale = size / 30
        thick = max(1, size // 15)
        for text, x, y, color in labels:
            text = to_ascii(text)
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            y = max(y, th + base + 4)
            cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
            cv2.putText(img, text, (x + 3, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_COLOR,
                        thick, cv2.LINE_AA)
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, x, y, color in labels:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        pad = max(2, size // 6)
        y0 = max(y - th - 2 * pad, 0)
        draw.rectangle([x, y0, x + tw + 2 * pad, y0 + th + 2 * pad], fill=_rgb(color))
        draw.text((x + pad - left, y0 + pad - top), text, font=font, fill=TEXT_COLOR)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def label_of(d):
    if d["name"] == "traffic light":
        return LIGHTS_VI[d.get("color", "unknown")]
    text = f"{NAMES_VI.get(d['name'], d['name'])} {d['conf']:.2f}".replace(".", ",")
    return text + (" · trong làn" if d.get("in_lane") else "")


def box_color(d):
    if d["name"] == "traffic light":
        return LIGHT_COLORS[d.get("color", "unknown")]
    if d.get("in_lane"):
        return BOX_IN_LANE
    return BOX_PERSON if d["name"] == "person" else BOX_VEHICLE


def draw(frame, ego, other, lane, dets, layers, banner=None):
    vis = frame.copy()
    h, w = vis.shape[:2]
    thick = max(2, round(min(h, w) / 300))
    size = max(14, round(min(h, w) / 40))

    if "other" in layers:
        _blend(vis, other, OTHER_COLOR, 0.40)
    if "ego" in layers:
        _blend(vis, ego, EGO_COLOR, 0.45)
    if "lines" in layers:
        _blend(vis, lane, LINE_COLOR, 0.85)

    labels = []
    if "boxes" in layers:
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            color = box_color(d)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
            labels.append((label_of(d), x1, y1, color))

    if banner:
        bar_h = int(size * 2)
        cv2.rectangle(vis, (0, h - bar_h), (w, h), BANNER_COLOR, -1)
        labels.append((" · ".join(banner), 0, h, BANNER_COLOR))

    return _draw_texts(vis, labels, size) if labels else vis
