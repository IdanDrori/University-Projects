import numpy as np


class Vector:
    """
    Represents a directed ray that starts at (0,0,0), with a direction
    Attributes:
        coords (np.ndarray): Coordinates of the vector's direction
        magnitude (float): Vector's magnitude
    """
    def __init__(self, coords: np.ndarray) -> None:
        """ Constructor for a Vector instance """
        self.coords = coords
        self.magnitude = np.linalg.norm(coords)

    def normalize(self) -> "Vector":
        """
        Returns a normalized vector
        Returns:
            Normalized vector
        """
        return Vector(self.coords / self.magnitude)

    def projection(self, vect2: "Vector") -> "Vector":
        """
        Projects the current vector onto another vector
        Args:
            vect2: Vector onto which the current vector will be projected

        Returns:
            Projected vector
        """
        return (np.dot(self.coords, vect2.coords) / np.dot(vect2.coords, vect2.coords)) * vect2.coords


class Ray:
    """
    Represents a directed vector that starts at a given origin point
    Attributes:
        origin (np.ndarray): Coordinates of ray's origin
        direction (Vector): Normalized direction vector
    """
    def __init__(self, origin, direction):
        """ Constructor for Ray instance """
        self.origin = origin
        self.direction = direction.normalize()

