from typing import Iterable, Union

from core.movement import Movement
from utils.track import detect_list_roatated_movement


class Box(dict):
    def __init__(self, x: float, y: float, w: float, h: float, angle: float):

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle

        super().__init__(self.__dict__)

    @property
    def box(self):
        return (self.x, self.y, self.w, self.h, self.angle)


class BoxDetect(Box):
    def __init__(
        self,
        id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        xl: float,
        yl: float,
        wm: float,
        hm: float,
        angle: float,
        cate_id: float = 0,
        score: float = 1.0,
    ) -> None:

        self.xl = xl
        self.yl = yl
        self.wm = wm
        self.hm = hm

        self.cate_id = int(cate_id)
        self.angle = angle
        self.score = score

        self._id = id
        self._im_path = ""
        self.went_by = False

        super().__init__(x, y, w, h, angle)
        self.__update()

    @property
    def im_path(self):
        return self._im_path

    @im_path.setter
    def im_path(self, v: str):
        self._im_path = v
        self.__update()

    def __update(self):
        dict.update(self, self.__dict__)

    def is_moved(
        self, target: "BoxDetect" | Iterable[Union[float, int]]
    ) -> None | Movement:
        """Check if box moved to target

        Args:
            target (BoxDetect | Iterable[Union[ float, int ]]): has same shape as self

        Returns:
            _type_: _description_
        """
        target_box = target_box if isinstance(target, "BoxDetect") else target
        if self.cate_id != target.cate_id:
            return None
        movements = detect_list_roatated_movement(
            [self.box], [target_box], translation_threshold=25, rotation_threshold=7
        )
        if len(movements):
            return movements[0]
        else:
            return None

    @staticmethod
    def detect_movement(box1: "BoxDetect", box2: "BoxDetect"):
        return box1.is_moved(box2)
