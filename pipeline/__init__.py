from typing import List


class UnitObject:
    def __init__(self, box=[], idx=None):
        """
        Create bounding box with id of vector
        :param bounds: [left, top, right, bottom]
        :param id: type of classification
        """
        self.box: List[int] = box
        self.class_id: int = idx

    def __str__(self):
        return str(self.class_id) + ": " + str(self.box)