import numpy as np

from scene import Scene
from light import Light
from ray import Ray, Vector
from material import Material
from color import Color
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane as Plane
from surfaces.sphere import Sphere
from intersection import Intersection, intersected_objects,nearest_intersection
from tqdm import tqdm


def render(scene: Scene, lights: list[Light], cubes: list[Cube], planes: list[Plane], spheres: list[Sphere]) -> np.ndarray:
    """
    Renders an image of a 3D scene by casting rays through each pixel of the screen.
    Function traces rays from the camera to the scene, computes intersections with objects
    (cubes, planes, spheres), and calculates the resulting colors based on lighting, material properties,
    and reflections. The final image is returned as a NumPy array of RGB values.

    Args:
        scene: The scene to render, including camera and other scene settings.
        lights: The list of lights that illuminate the scene.
        cubes: The list of cube objects in the scene.
        planes: The list of plane objects in the scene.
        spheres: The list of sphere objects in the scene.

    Returns:
        A 3D NumPy array representing the rendered image with shape (height, width, 3),
        where each pixel contains RGB color values.
    """

    width = scene.width
    height = scene.height
    origin = scene.camera.position

    count = 0
    image = np.zeros((height, width, 3))
    for i, y in tqdm(enumerate(np.linspace(0, scene.screen_height, height))):
        for j, x in enumerate(np.linspace(0, scene.screen_width, width)):
            count = count + 1

            # p = the current pixel in the screen
            p = scene.p0 + (scene.camera_right_vect.coords * x) + (scene.screen_up_vector().coords * y)

            # ray = the vector cast from the camera through p in the look-at direction
            ray = Ray(origin, Vector(p - origin))

            # inters = a list of surfaces that intersect with ray sorted by the distance from the camera
            inters = intersected_objects(cubes, planes, spheres, ray)

            image[i][j] = colors_rec(lights, inters, ray, scene, cubes, planes, spheres,
                                     int(scene.settings.max_recursions)).rgb

    return image


def get_color(light: Light, intersect: Intersection, ray: Ray, material: Material, light_intensity: float) -> Color:
    """
    Computes the color at an intersection point based on diffuse and specular lighting.
    This function calculates the final color at the intersection point by combining diffuse
    and specular components based on the material properties, light intensity, and reflection direction.

    Args:
        light: The light source used for calculating the color.
        intersect: The intersection of the ray with an object in the scene.
        ray: The ray that intersects the object.
        material: The material properties of the intersected surface.
        light_intensity: The intensity of the light source affecting the intersection.

    Returns:
        The resulting color at the intersection, considering diffuse and specular lighting.
    """
    light_dir = Vector(light.position - intersect.point).normalize()
    light_reflection_dir = \
        Vector(light_dir.coords + 2 * (light_dir.projection(Vector(intersect.normal)) - light_dir.coords)).normalize()
    diffuse_color = Color.diffuse(material.diffuse_color, light.color, Vector(intersect.normal), light_dir,
                                  light_intensity)
    specular_color = Color.specular(material.specular_color, light.color, light.specular_intensity,
                                    light_reflection_dir, ray.direction, material.shininess, light_intensity)
    return Color(diffuse_color.rgb + specular_color.rgb)


def colors_rec(lights: list[Light], intersections: list[Intersection], ray: Ray, scene: Scene, cubes: list[Cube],
               planes: list[Plane], spheres: list[Sphere], max_recursion: int) -> Color:
    """
    Recursively computes the color at a given list of intersections, considering lighting, transparency, and reflection.
    The function calculates the color of a surface at a given intersection by considering multiple
    factors, including diffuse and specular lighting, transparency for semi-transparent objects, and
    reflections. If recursion depth allows, it also handles reflection of light from the surface.

    Args:
        lights: The list of light sources that affect the scene.
        intersections : The list of intersections, sorted by distance from the camera.
        ray: The ray from the camera or reflection that intersects the scene.
        scene: The scene that contains the objects, materials, and settings.
        cubes: The list of cubes in the scene.
        planes: The list of planes in the scene.
        spheres: The list of spheres in the scene.
        max_recursion: The maximum recursion depth for reflection calculations.

    Returns:
        The computed color at the intersection point, including lighting, transparency, and reflections.
    """
    background = scene.settings.background_color
    if len(intersections) == 0:
        return background
    mat = scene.materials
    intersect = intersections[0]
    index = intersect.surface.material_index - 1
    material = mat[int(index)]
    transparency = material.transparency


    color_lighting = calculate_lighting(intersect, material, lights, scene, ray, cubes, planes, spheres)
    # Calculate background color for transparent and semi-transparent surfaces
    if transparency > 0 and len(intersections) > 1:
        background = colors_rec(lights, intersections[1:], ray, scene, cubes, planes, spheres, max_recursion)

    # Calculate reflections of the surface
    reflection_color = Color(np.array([0, 0, 0]))
    if max_recursion > 0:
        reflection_dir = Vector(-ray.direction.coords)
        reflection_vector = Vector(reflection_dir.coords + 2 *
                                   (reflection_dir.projection(Vector(intersect.normal)) -
                                    reflection_dir.coords)).normalize()
        reflection_ray = Ray(intersect.point, reflection_vector)
        ref_intersections = intersected_objects(cubes, planes, spheres, reflection_ray)
        reflection_color = colors_rec(lights, ref_intersections, reflection_ray, scene, cubes, planes, spheres,
                                      max_recursion - 1)

    return Color.output_color(background, transparency, color_lighting, reflection_color, material.reflection_color)


def calculate_lighting(intersect: Intersection, material: Material, lights: list[Light], scene: Scene, ray: Ray,
                       cubes: list[Cube], planes: list[Plane], spheres: list[Sphere]) -> Color:
    """

    Args:
        intersect: intersection point
        material: matirial at intersection
        lights: The list of light sources that affect the scene.
        scene: The scene that contains the objects, materials, and settings.
        ray: The ray from the camera
        cubes: The list of cubes in the scene.
        planes: The list of planes in the scene.
        spheres: The list of spheres in the scene.

    Returns:
        Color of the light at intersection
    """
    final_color = np.array([0, 0, 0],dtype='float64')
    intesity = 0
    for ls in lights:
        precentage = compute_shadow(scene.settings.root_number_shadow_rays,ls,intersect,cubes,planes,spheres)
        intesity += 1-ls.shadow_intensity + ls.shadow_intensity*precentage
        final_color += get_color(ls,intersect,ray,material,1).rgb
    return Color(final_color*(intesity/len(lights)))



def compute_shadow(rays: int, light: Light, intersect: Intersection, cubes: list[Cube], planes: list[Plane], spheres: list[Sphere]) -> float:
    if rays == 0:
        return 1.0
    rays = int(rays)
    main_dir = intersect.point - light.position
    dist_to_surface = np.linalg.norm(main_dir)
    if dist_to_surface < 1e-9:
        return 1.0  # Fully lit if the surface is extremely close to the light
    main_dir /= dist_to_surface  # Normalize

    # Build a small coordinate frame perpendicular to main_dir
    e1 = np.array([1, 0, 0], dtype=float)
    if abs(main_dir @ e1) > 0.99:
        e1 = np.array([0, 1, 0], dtype=float)
    right = np.cross(main_dir, e1)
    right /= np.linalg.norm(right)
    up = np.cross(main_dir, right)
    up /= np.linalg.norm(up)
    radius = light.radius


    # Generate a grid of random points within sub-squares
    grid_indices = np.indices((rays, rays)).reshape(2, -1).T  # Array of (row, col) indices
    subsquare_size = radius / rays  # Size of each sub-square
    
    # Calculate base positions for sub-squares
    base_positions = light.position - radius / 2 * right - radius / 2 * up + \
                     grid_indices[:, 0][:, None] * subsquare_size * right + \
                     grid_indices[:, 1][:, None] * subsquare_size * up
    # Add random offsets within each sub-square
    rand_offsets = np.random.rand(rays * rays, 2)  # Random offsets in range [0, 1]
    light_grid = base_positions + rand_offsets[:, 0][:, None] * subsquare_size * right + \
                 rand_offsets[:, 1][:, None] * subsquare_size * up

    directions = intersect.point - light_grid
    distances = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= distances  # Normalize directions

    shadow_rays = [Ray(light_grid[i], Vector(directions[i])) for i in range(light_grid.shape[0])]
    intersections = [nearest_intersection(cubes, planes, spheres, ray) for ray in shadow_rays]

    hits = sum(
        hit is None or np.linalg.norm(hit.point - light_grid[i]) >= distances[i, 0] - 1e-5
        for i, hit in enumerate(intersections)
    )
    return hits / (rays * rays)
