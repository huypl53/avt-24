from core.box_detect import BoxDetect
from typing import Iterable, Union, List


class BoxRecord:
    def __init__(self, steps_num: int, *args) -> None:
        super().__init__(*args)
        self.steps_num = steps_num
        self.last: BoxDetect | None = None
        self.history: List[BoxDetect | None] = [None] * (steps_num - 1)
        self.records: List[BoxDetect | None] = [None] * steps_num

    def detect_movement(
        self, target: BoxDetect | Iterable[float | int], step: int, save=False
    ):
        if not isinstance(target, BoxDetect):
            target = BoxDetect(*target)

        self.history[step] = target
        self.last = target
        movement = BoxDetect.detect_movement(self, target)
        if step == 0:
            return movement

        if save:
            self.store(step, target)
        return movement

    def store(self, step: int, target: "BoxDetect"):
        assert step < self.steps_num
        self.records[step] = target

        if step == 0:
            pass
        else:
            self.history[step - 1]
