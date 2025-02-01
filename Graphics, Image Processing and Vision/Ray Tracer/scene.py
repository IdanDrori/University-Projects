import numpy as np
from camera import Camera
from material import Material
from scene_settings import SceneSettings
from ray import Vector


class Scene:
    # Represents the objects of a scene, and the calculations of the screen and camera
    def __init__(self, camera, materials, settings, width, height):
        """
        Init function
        Args:
            camera (Camera):
            materials (list[Material]):
            settings (SceneSettings):
            width (int):
            height (int):
        """

        self.camera = camera
        self.materials = materials
        self.settings = settings
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.screen_width = self.camera.screen_width
        self.screen_height = self.screen_width / self.aspect_ratio
        # Vector of the camera's look-at direction towards the scene
        self.towards = Vector(self.camera.look_at.coords - self.camera.position).normalize()
        # Screen center is the point at the center of the screen, by the vector towards
        self.screen_center = self.camera.position + (self.camera.screen_distance * self.towards.coords)
        # The vector that points to the right of the camera
        self.camera_right_vect = Vector(np.cross(self.screen_up_vector().coords, -self.towards.coords))
        # p0 is the point in the lower left corner of the screen, that's the point we begin rendering rays from
        self.p0 = self.screen_center - (self.screen_width / 2 * self.camera_right_vect.coords) \
                  - (self.screen_height / 2 * self.screen_up_vector().coords)

    def screen_up_vector(self):
        """
        Computes the normalized vector that's perpendicular to self.towards, and aligns with the camera's up vector.
        Ensures the screen is properly oriented relative to the camera.
        Returns:
            Vector: The vector perpendicular to self.towards and aligns with the camera's up vector.
        """
        up_vector = self.camera.up_vector
        towards = self.towards
        proj = up_vector.projection(towards)
        if proj.any():  # If proj contains any non-zero elements
            return Vector(-1 * (up_vector.coords - proj)).normalize()
        else:
            return up_vector.normalize()