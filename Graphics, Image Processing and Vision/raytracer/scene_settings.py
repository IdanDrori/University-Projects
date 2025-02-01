import numpy as np
from color import Color


class SceneSettings:
    def __init__(self, background_color, root_number_shadow_rays, max_recursions):
        self.background_color = Color(np.array(background_color))
        self.root_number_shadow_rays = root_number_shadow_rays
        self.max_recursions = max_recursions
