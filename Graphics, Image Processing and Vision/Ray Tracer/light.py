import numpy as np
from color import Color


class Light:
    """
    Represent a light source, includes the position the light source, its color, specular intensity,
    shadow intensity and radius.
    Attributes:
        position (np.ndarray): Coordinates of the light's position
        color (Color): The light's color
        specular_intensity (float): Specular intensity
        shadow_intensity (float): Shadow intensity
        radius (float): Radius of the light
    """
    def __init__(self, position: list[float], color: list[float],
                 specular_intensity: float, shadow_intensity: float, radius: float) -> None:
        """ Constructor for a Light instance """
        self.position = np.array(position)
        self.color = Color(np.array(color))
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius
