import cv2
import numpy as np

# Loai bo nhieu
def _remove_small(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8
    )

    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area

    return keep[labels]

# Tim pixel duong gan tam nhat
def _nearest_true(row, x, max_dist):
    xs = np.flatnonzero(row)

    if xs.size == 0:
        return None

    i = int(xs[np.argmin(np.abs(xs - x))])

    return i if abs(i - x) <= max_dist else None

# Tim doan lien tuc chua tam
def _run_around(row, x):
    left_false = np.flatnonzero(~row[:x])
    right_false = np.flatnonzero(~row[x:])

    l = int(left_false[-1]) + 1 if left_false.size else 0
    r = x + int(right_false[0]) - 1 if right_false.size else row.shape[0] - 1

    return l, r


def split_road(drivable, lane, ego_width_bottom=0.45):
    H, W = drivable.shape
    drivable = drivable.astype(bool)
    lane = _remove_small(
        lane.astype(bool),
        min_area=max(20, H * W // 4000)
    )
    road = drivable | lane
    ego = np.zeros((H, W), dtype=bool)

    rows = np.flatnonzero(drivable.any(axis=1))

    if rows.size == 0:
        return ego, drivable
    
    y_top, y_bot = int(rows.min()), int(rows.max())

    cx = W // 2
    last_w = last_y = None
    for y in range(y_bot, y_top - 1, -1):
        row = road[y]
        x0 = cx if row[cx] else _nearest_true(row, cx, W // 10)
        if x0 is None:
            continue
        rl, rr = _run_around(row, x0)

        span = max(y - y_top, 1)
        if last_w is not None:
            exp_w = last_w * span / max(last_y - y_top, 1)
        else:
            exp_w = ego_width_bottom * W * span / max(H - 1 - y_top, 1)

        xs = np.flatnonzero(lane[y, rl:rr + 1]) + rl
        left, right = xs[xs < x0], xs[xs > x0]
        xl = int(left.max()) + 1 if left.size else None
        xr = int(right.min()) - 1 if right.size else None

        if xl is not None and xr is not None:
            if last_w is None:
                plausible = xr - xl >= 0.08 * W
            else:
                plausible = 0.5 * exp_w <= xr - xl <= 1.8 * exp_w
        else:
            plausible = True
        if not plausible:
            if abs((x0 - xl) - exp_w / 2) <= abs((xr - x0) - exp_w / 2):
                xr = None
            else:
                xl = None

        if xl is not None and xr is not None:
            last_w, last_y = xr - xl, y
        elif xl is not None:
            xr = int(xl + exp_w)
        elif xr is not None:
            xl = int(xr - exp_w)
        else:
            xl, xr = int(x0 - exp_w / 2), int(x0 + exp_w / 2)

        xl, xr = max(xl, rl), min(xr, rr)
        if xr - xl < 2:
            continue
        ego[y, xl:xr + 1] = drivable[y, xl:xr + 1]
        cx = (xl + xr) // 2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ego_u8 = cv2.morphologyEx(ego.astype(np.uint8), cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ego_u8, connectivity=8)
    if n > 1:
        ego = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        ego = ego_u8.astype(bool)
    return ego, drivable & ~ego

