import warnings
import cv2
import numpy as np
import torch

IMG_W, IMG_H, PAD = 640, 360, 12

class RoadSegmenter:
    def __init__(self, weights="weights/yolopv2.pt", device="cuda:0"):
        self.device = torch.device(device)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.model = torch.jit.load(weights, map_location=self.device)

        self.half = self.device.type == "cuda"

        if self.half:
            self.model.half()

        self.model.eval()

    @torch.no_grad()
    def __call__(self, frame_bgr):
        img = cv2.resize(frame_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, PAD, PAD, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))

        t = torch.from_numpy(img).to(self.device)
        t = (t.half() if self.half else t.float()) / 255.0

        out = self.model(t.unsqueeze(0))
        seg, ll = out[1], out[2]

        seg = seg[:, :, PAD:PAD + IMG_H, :]
        ll = ll[:, :, PAD:PAD + IMG_H, :]

        drivable = seg.argmax(1)[0].cpu().numpy().astype(bool)
        lane = (ll[0, 0] > 0.5).cpu().numpy()

        return drivable, lane
