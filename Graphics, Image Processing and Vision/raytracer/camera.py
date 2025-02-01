import numpy as np
from ray import Vector


class Camera:
    """
    Represents a camera object, with position, orientation and screen properties.
    Attributes:
        position (np.ndarray): The position of the camera in 3D space.
        look_at (Vector): The point the camera is looking at.
        up_vector (Vector): The upward direction of the camera.
        screen_distance (float): The distance from the camera to the screen.
        screen_width (float): The width of the screen.
    """
    def __init__(self, position: list[float], look_at: list[float],
                 up_vector: list[float], screen_distance: float, screen_width: float) -> None:
        """ Constructor for a Camera instance """
        self.position = np.array(position)
        self.look_at = Vector(np.array(look_at))
        self.up_vector = Vector(np.array(up_vector))
        self.screen_distance = screen_distance
        self.screen_width = screen_width
