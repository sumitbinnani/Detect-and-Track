class UnitObject:
    def __init__(self, box=[], idx=None):
        """
        Create bounding box with id of vector
        :param bounds: [left, top, right, bottom]
        :param id: type of classification
        """
        self.box = box
        self.idx = idx

    def __str__(self):
        return str(self.idx) + ": " + str(self.box)