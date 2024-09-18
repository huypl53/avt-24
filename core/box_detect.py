from typing import Iterable, Union

from utils.track import detect_movement_rotated


class BoxDetect:
    def __init__(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        # xl: float,
        # yl: float,
        # wm: float,
        # hm: float,
        angle: float,
        cate_id: float = 0,
        score: float = 1.0,
    ) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        # self.xl = xl
        # self.yl = yl
        # self.wm = wm
        # self.hm = hm

        self.cate_id = int(cate_id)
        self.angle = angle
        self.score = score
        self.im_id = ""
        self.im_path = ""

    @property
    def box(self):
        return (self.x, self.x, self.w, self.h, self.angle)

    def is_moved(self, target: "BoxDetect" | Iterable[Union[float, int]]):
        """Check if box moved to target

        Args:
            target (BoxDetect&#39; | Iterable[Union[ float, int ]]): has same shape as self
        """
        if isinstance(target, "BoxDetect"):
            return detect_movement_rotated([self.box], [target.box])
        else:
            return detect_movement_rotated([self.box], [target])

    @classmethod
    def detect_movement(cls, box1: "BoxDetect", box2: "BoxDetect"):
        pass
