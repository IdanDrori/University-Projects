import numpy as np


class InfinitePlane:
    """
    Represents an infinite plane with normal, offset and material index.
    Attributes:
        normal (np.ndarray): Coordinates of the normal vector to the plane.
        offset (float): Offset of the plane.
        material_index (int): Index of the material assigned to the plane.
    """
    def __init__(self, normal: list[float], offset: float, material_index: int) -> None:
        """ Constructor for an Infinite Plane instance """
        self.normal = np.array(normal)
        self.offset = offset
        self.material_index = material_index
