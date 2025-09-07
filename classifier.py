from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# 			        Chiều dài 	Chiều rộng 	ratio = dài/rộng
# Tàu cá 			18			4			10
# Tàu quân sự		120			8			12
# Tàu hàng			200			15			10
# Sân bay			1000		15			50
# Trực thăng		15			10			1.5
# trận địa			100			100			1


@dataclass
class Target:
    width: int
    height: int
    ratio: float
    name: str


TARGET_LIST = [
    Target(width=4, height=18, ratio=10, name="tau_ca"),
    Target(width=8, height=120, ratio=12, name="tau_quan_su"),
    Target(width=15, height=200, ratio=10, name="tau_hang"),
    Target(width=15, height=1000, ratio=50, name="san_bay"),
    Target(width=10, height=15, ratio=1.5, name="truc_thang"),
    Target(width=100, height=100, ratio=1, name="tran_dia"),
]

TG_W = [2, 2, 1]


def spatial_classify(w: float, h: float) -> Tuple[Target, float]:
    w, h = (h, w) if w > h else (w, h)
    d: List[Tuple[float, float, float]] = []
    ratio = max(w, h) / min(w, h)

    for i, target in enumerate(TARGET_LIST):
        t_w, t_h, t_r = target.width, target.height, target.ratio
        d.append((abs(w - t_w), abs(h - t_h), abs(ratio - t_r)))
        # print(target.name, d[-1])

    w_d = np.array(d)
    w_d = w_d / np.sum(w_d, axis=0)
    w_d = np.average(w_d, weights=TG_W, axis=1)
    target_i = np.argmin(w_d)
    score = 1 - w_d[target_i]
    return TARGET_LIST[target_i], score
