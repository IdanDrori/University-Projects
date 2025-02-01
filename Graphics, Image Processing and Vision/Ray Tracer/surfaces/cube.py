import numpy as np


class Cube:
    """
    Represents a 3D cube with position (center of the cube), scale and material index
    Attributes:
        position (np.ndarray): Coordinates of the center of the cube
        scale (float): Scale factor of the cube
        material_index (int): Index of the material assigned to the cube
    """
    def __init__(self, position: list[float], scale: float, material_index: int) -> None:
        """ Constructor for a Cube instance """
        self.position = np.array(position)
        self.scale = scale
        self.material_index = material_index
