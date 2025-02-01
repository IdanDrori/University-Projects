import numpy as np


class Sphere:
    """
        Represents a sphere with position, radius and material index.
        Attributes:
            position (np.ndarray): Coordinates of the center of the sphere.
            radius (float): Radius of the sphere.
            material_index (int): Index of the material assigned to the sphere.
        """
    def __init__(self, position: list[float], radius: float, material_index: int) -> None:
        """ Constructor for Sphere instance """
        self.position = np.array(position)
        self.radius = radius
        self.material_index = material_index
