import time

import cv2
import numpy as np

from .ego_lane import split_road
from .object_detection import VEHICLES
from .visualize import draw

ALL_LAYERS = {"ego", "other", "lines", "boxes"}


def _resize_mask(mask, w, h):
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5


def _in_mask(mask, box):
    h, w = mask.shape
    x1, _, x2, y2 = box
    cx = min(max((x1 + x2) // 2, 0), w - 1)
    dx = max(2, (x2 - x1) // 10)
    y_lo, y_hi = max(y2 - 4, 0), min(y2 + max(4, h // 50), h)
    return bool(mask[y_lo:y_hi, max(cx - dx, 0):cx + dx + 1].any())


class ADASPipeline:
    def __init__(self, yolo_weights="weights/yolov8s.pt", yolopv2_weights="weights/yolopv2.pt", device=None,
                 light_cls=None):
        import torch

        from .object_detection import ObjectDetector
        from .road_detection import RoadSegmenter

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.objects = ObjectDetector(yolo_weights, device, light_cls)
        self.light_method = "cls" if light_cls else "hsv"
        self.road = RoadSegmenter(yolopv2_weights, device)
        self._ego_acc = None

    def warmup(self):
        self.process(np.zeros((720, 1280, 3), np.uint8))
        self.reset()

    def reset(self):
        self._ego_acc = None

    def process(self, frame, conf=0.35, layers=ALL_LAYERS, smooth=False):
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        drivable, lane = self.road(frame)
        ego, other = split_road(drivable, lane)
        if smooth:
            e = ego.astype(np.float32)
            self._ego_acc = e if self._ego_acc is None or self._ego_acc.shape != e.shape else 0.6 * self._ego_acc + 0.4 * e
            ego = self._ego_acc > 0.5
            other = drivable & ~ego
        ego_f, other_f, lane_f = (_resize_mask(m, w, h) for m in (ego, other, lane))

        dets = [d for d in self.objects(frame, conf)
                if not (d["box"][3] > 0.95 * h and d["box"][2] - d["box"][0] > 0.5 * w)]
        for d in dets:
            d["in_lane"] = d["name"] != "traffic light" and _in_mask(ego_f, d["box"])

        lights = [d for d in dets if d["name"] == "traffic light" and d.get("color", "unknown") != "unknown"]
        light = None
        if lights:
            big = max(lights, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
            light = big["color"]

        persons_in = [d for d in dets if d["in_lane"] and d["name"] == "person"]
        vehicles_in = [d for d in dets if d["in_lane"] and d["name"] in VEHICLES]
        alerts = []
        if light == "red":
            alerts.append(("danger", "Đèn đỏ phía trước", "RED LIGHT"))
        elif light == "yellow":
            alerts.append(("warning", "Đèn vàng, sắp chuyển đỏ", "YELLOW LIGHT"))
        if persons_in:
            alerts.append(("danger", f"{len(persons_in)} người trong làn đang chạy", "PERSON IN LANE"))
        if vehicles_in:
            alerts.append(("warning", f"{len(vehicles_in)} xe trong làn đang chạy", "VEHICLE IN LANE"))

        vis = draw(frame, ego_f, other_f, lane_f, dets, layers, banner=[a[1] for a in alerts])
        info = {
            "alerts": alerts,
            "light": light,
            "in_lane": persons_in + vehicles_in,
            "n_vehicles": sum(d["name"] in VEHICLES for d in dets),
            "n_persons": sum(d["name"] == "person" for d in dets),
            "ms": (time.perf_counter() - t0) * 1000,
        }
        return vis, info