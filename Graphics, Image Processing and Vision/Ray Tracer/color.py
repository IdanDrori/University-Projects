import numpy as np
from ray import Vector


class Color:
    """
    Represents a color by a numpy ndarray.
    Attributes
        rgb (np.ndarray): Numpy array of RGB values.
    """
    def __init__(self, rgb: np.ndarray) -> None:
        """ Constructor for a Color instance """
        self.rgb = rgb

    @staticmethod
    def diffuse(material_diffuse: "Color", light_color: "Color", normal: Vector, light_dir: Vector,
                intensity: float) -> "Color":
        """
        Calculates diffuse color of a given material with a given light source.
        Args:
            material_diffuse: Diffuse color of the material
            light_color: Color of the light ray
            normal: Normal of the intersection point
            light_dir: Vector of light ray
            intensity: Intensity value
        Returns:
            Final color
        """

        return Color(intensity * material_diffuse.rgb * light_color.rgb * max(0.0, np.dot(normal.coords, light_dir.coords)))

    @staticmethod
    def specular(material_specular: "Color", light_color: "Color", light_specular: float, reflect_dir: Vector,
                 ray_dir: Vector, shininess: float, intensity: float) -> "Color":
        """
        Calculates the specular color of a given material with a given light source
        Args:
            material_specular: Specular color of the material
            light_color: Color of the light ray
            light_specular: Specular intensity value of the light
            reflect_dir: Vector of reflected ray
            ray_dir: Vector of light ray
            shininess: Shininess value (phong)
            intensity: Intensity value
        Returns:
            Color: Final color
        """

        return Color(intensity * material_specular.rgb * light_color.rgb * light_specular *
                     (max(0.0, np.dot(reflect_dir.coords, -ray_dir.coords)) ** shininess))

    @staticmethod
    def output_color(background: "Color", transparency: float, light_color: "Color", ref: "Color",
                     material_ref: "Color") -> "Color":
        """
        Calculates the output color of a given material being hit by a given light ray
        Args:
            background: Background color
            transparency: Transparency value
            light_color: Light color
            ref: Reflected color
            material_ref: Material's reflective color
        Returns:
            Final color
        """
        return Color(np.clip((background.rgb * transparency) + light_color.rgb * (1 - transparency) +
                     (ref.rgb * material_ref.rgb),0,1))

