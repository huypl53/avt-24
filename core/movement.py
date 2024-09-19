from typing import List
from core.box_detect import BoxDetect


class Movement(dict):
    def __init__(self, displacement: float, rotaion_shift: float) -> None:
        self.displacement = displacement
        self.rotation_shift = rotaion_shift
        # dict.__init__(self, displacement=displacement, rotation_shift=rotaion_shift)
        # mapping = {"displacement": displacement, "rotation_shift": rotaion_shift}
        mapping = self.__dict__
        super().__init__(mapping)


class BoxMovement(Movement):
    def __init__(
        self,
        pre_box: BoxDetect,
        next_box: BoxDetect,
        displacement: float,
        rotaion_shift: float,
    ) -> None:
        self.pre_box = pre_box
        self.next_box = next_box
        super().__init__(displacement, rotaion_shift)


if __name__ == "__main__":
    pass
