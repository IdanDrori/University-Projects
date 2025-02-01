import numpy as np
from color import Color


class Material:
    """
    Represents the material properties used for shading and rendering, including colors for diffuse,
    specular, and reflection, along with shininess and transparency factors.

    Attributes:
        diffuse_color (Color): The diffuse color of the material.
        specular_color (Color): The specular color of the material.
        reflection_color (Color): The reflection color of the material.
        shininess (float): The shininess factor for the material (phong).
        transparency (float): The transparency level of the material (0.0 = opaque, 1.0 = fully transparent).
    """
    def __init__(self, diffuse_color: list[float], specular_color: list[float], reflection_color: list[float],
                 shininess: float, transparency: float) -> None:
        """ Constructor for a Material instance """
        self.diffuse_color = Color(np.array(diffuse_color))
        self.specular_color = Color(np.array(specular_color))
        self.reflection_color = Color(np.array(reflection_color))
        self.shininess = shininess
        self.transparency = transparency
