from typing import List

from core.box_detect import BoxDetect
from core.movement import Movement


class BoxRecord(dict):
    def __init__(
        self,
        cate_id: int,
        steps_num: int,
        start_step: int = 0,
    ) -> None:
        self.cate_id = cate_id
        self.steps_num = steps_num
        self.last_step: int = 0
        self.history: List[Movement | None] = [None] * (steps_num - 1)
        self.longest_history: List[Movement] = []

        self.records: List[BoxDetect | None] = [None] * steps_num
        self.start_step = start_step
        self.longest_sequence: List[BoxDetect] = []
        self.max_start_i, self.max_end_i = 0, 0
        self.__update()

    def check_new_target(self, target: BoxDetect, step: int, save=True) -> bool:
        _last_record = self.last_record

        if self.last_step == 0 or step > self.last_step:
            self.last_step = step
        if not _last_record:
            if save:
                self.records[step] = target
            return True

        movement = BoxDetect.detect_movement(_last_record, target)
        if save:
            self.history[step] = movement
            self.records[step] = target

        if not movement:
            return False
        return True

    def __update(self):
        dict.update(self, self.__dict__)

    @property
    def last_record(self):
        try:
            return self.records[self.last_step]
        except:
            return None

    def update_longest_sequence(self):
        start_i, max_start_i = 0, 0
        max_len = 0
        current_len = 0
        for i in range(self.steps_num):
            record = self.records[i]
            if record is not None:
                current_len += 1
                if current_len > max_len:
                    max_len = current_len
                    max_start_i = start_i
            else:
                current_len = 0
                start_i = i + 1
        max_end_i = max_start_i + max_len - 1
        if max_len < 1:
            return
        self.max_start_i = max_start_i
        self.max_end_i = max_end_i
        self.longest_sequence = [r for r in self.records[max_start_i:max_end_i] if r]
        self.longest_history = [
            h for h in self.history[max_start_i : max_end_i - 1] if h
        ]
        self.__update()
