import numpy as np
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane as Plane
from surfaces.sphere import Sphere
from ray import Ray, Vector


class Intersection:
    """
    Represents an intersection, includes which surface the intersection is on, the intersection point,
    the normal at that point, and the distance of the intersection from the camera.
    Attributes:
        surface (Cube | Sphere | Plane): The surface where the intersection occurs.
        point (np.ndarray): The 3D coordinates of the intersection point.
        normal (np.ndarray): The normal vector at the intersection point.
        distance (float): The distance from the camera to the intersection point.
    """
    def __init__(self, surface: Cube | Sphere | Plane, point: np.ndarray, normal: np.ndarray, distance: float) -> None:
        """ Constructor for an Intersection instance """
        self.surface = surface
        self.point = point
        self.normal = normal
        self.distance = distance


def cube_intersect(cube: Cube, ray: Ray) -> Intersection | None:
    """
    Returns the intersection point between a given cube and ray. Returns None if there's no intersection.
    Args:
        cube: Relevant Cube object
        ray: Relevant Ray object

    Returns:
        Intersection point, or None if there's no intersection
    """
    position, scale = cube.position, cube.scale
    origin, direction = ray.origin, ray.direction.coords

    x_min, x_max = position[0] - scale / 2, position[0] + scale / 2
    y_min, y_max = position[1] - scale / 2, position[1] + scale / 2
    z_min, z_max = position[2] - scale / 2, position[2] + scale / 2
    t_x_min, t_x_max = (x_min - origin[0]) / direction[0], (x_max - origin[0]) / direction[0]
    t_y_min, t_y_max = (y_min - origin[1]) / direction[1], (y_max - origin[1]) / direction[1]
    t_z_min, t_z_max = (z_min - origin[2]) / direction[2], (z_max - origin[2]) / direction[2]

    if t_x_max < t_x_min:
        t_x_min, t_x_max = swap(t_x_min, t_x_max)

    if t_y_max < t_y_min:
        t_y_min, t_y_max = swap(t_y_min, t_y_max)

    if t_z_max < t_z_min:
        t_z_min, t_z_max = swap(t_z_min, t_z_max)

    if (t_x_min > t_y_max) or (t_y_min > t_x_max):
        return None

    if t_y_min > t_x_min:
        t_x_min = t_y_min

    if t_y_max < t_x_max:
        t_x_max = t_y_max

    if (t_x_min > t_z_max) or (t_z_min > t_x_max):
        return None

    intersect_point = ray.origin + ray.direction.coords * t_x_min
    normal = normal_cube_intersect(intersect_point, cube.position, scale)

    return Intersection(cube, intersect_point, normal, t_x_min)


def plane_intersect(plane: Plane, ray: Ray) -> Intersection | None:
    """
    Returns the intersection point between a given plane and ray. Returns None if there's no intersection.
    Args:
        plane: Relevant InfinitePlane object
        ray: Relevant Ray object

    Returns:
        Intersection point between the plane and the ray, or None
    """

    normal = plane.normal
    offset = plane.offset
    denominator = np.dot(normal, ray.direction.coords)  # normal * (origin + t * direction) = offset
    t = (offset - np.dot(normal, ray.origin)) / denominator

    if t > 0:   # If there's an intersection
        intersect_point = ray.origin + ray.direction.coords * t
        return Intersection(plane, intersect_point, normal, t)

    return None  # Else there's no intersection


def sphere_intersect(sphere: Sphere, ray: Ray) -> Intersection | None:
    """
    Returns the intersection point of a given sphere and ray. Returns None if there's no intersection
    Args:
        sphere: Relevant Sphere object
        ray: Relevant Ray object

    Returns:
        Intersection point between sphere and ray, or None
    """
    position = sphere.position
    radius = sphere.radius

    b = 2 * np.dot(ray.direction.coords, ray.origin - position)
    c = np.linalg.norm(ray.origin - position) ** 2 - radius ** 2

    determinant = b ** 2 - 4 * c  # a = 1
    if determinant > 0:  # There's an intersection point if the determinant is larger than 0
        t1 = (-b + np.sqrt(determinant)) / 2
        t2 = (-b - np.sqrt(determinant)) / 2
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
            intersection_point = ray.origin + ray.direction.coords * t
            normal = Vector(intersection_point - position).normalize().coords
            return Intersection(sphere, intersection_point, normal, t)

    return None  # There's no intersection


def intersected_objects(cubes: list[Cube], planes: list[Plane], spheres: list[Sphere], ray: Ray) -> list[Intersection]:
    """
    Given lists of cubes, planes and spheres, ordered by distance, and given a ray,
    calculates all surfaces the ray intersects with. Function assumes that no 2 distances are the same
    Args:
        cubes: List of Cubes in the scene
        planes: List of Planes in the scene
        spheres: List of Spheres in the scene
        ray: relevant Ray

    Returns:
        List of intersection points
    """

    objects_inters = []
    if len(cubes) != 0:
        for cube in cubes:
            cube_inter = cube_intersect(cube, ray)
            if cube_inter is not None:
                objects_inters.append(cube_inter)

    if len(planes) != 0:
        for plane in planes:
            plane_inter = plane_intersect(plane, ray)
            if plane_inter is not None:
                objects_inters.append(plane_inter)

    if len(spheres) != 0:
        for sphere in spheres:
            sphere_inter = sphere_intersect(sphere, ray)
            if sphere_inter is not None:
                objects_inters.append(sphere_inter)

    objects_inters.sort(key=lambda x: x.distance)
    return objects_inters


def nearest_intersection(cubes: list[Cube], planes: list[Plane], spheres: list[Sphere], ray: Ray) -> Intersection:
    """
    Simple auxiliary function.
    Given list of Cubes, Planes and Spheres in the scene, and given a Ray, returns the closest intersection point
    Args:
        cubes: List of Cubes in the scene
        planes: List of Planes in the scene
        spheres: List of Spheres in the scene
        ray: Relevant Ray

    Returns:
        Nearest Intersection
    """
    nearest_object = intersected_objects(cubes, planes, spheres, ray)[0]
    return nearest_object


def normal_cube_intersect(inter_point: np.ndarray, position: np.ndarray, scale: float) -> np.ndarray:
    """
    Returns the normal of the intersection with a cube, according to which face the intersection is on.
    Args:
        inter_point: NDArray of intersection point coordinates
        position: Center of the cube
        scale: Scale

    Returns:
        Normal of intersection point, as NDArray.

    """
    EPS = 0.000001
    normal = np.empty(3, )
    near_point = np.array([position[0] - scale / 2, position[1] - scale / 2, position[2] - scale / 2])
    far_point = np.array([position[0] + scale / 2, position[1] + scale / 2, position[2] + scale / 2])
    if abs(inter_point[0] - near_point[0]) <= EPS:
        normal = np.array([-1, 0, 0], dtype=float)
    elif abs(inter_point[0] - far_point[0]) <= EPS:
        normal = np.array([1, 0, 0], dtype=float)
    elif abs(inter_point[1] - near_point[1]) <= EPS:
        normal = np.array([0, -1, 0], dtype=float)
    elif abs(inter_point[1] - far_point[1]) <= EPS:
        normal = np.array([0, 1, 0], dtype=float)
    elif abs(inter_point[2] - near_point[2]) <= EPS:
        normal = np.array([0, 0, -1], dtype=float)
    elif abs(inter_point[2] - far_point[2]) <= EPS:
        normal = np.array([0, 0, 1], dtype=float)

    return normal


def swap(val_a: float, val_b: float) -> tuple[float, float]:
    """
    Auxiliary function to swap the values of two variables
    Args:
        val_a: First value
        val_b: Second value

    Returns:
        The same values swapped
    """
    temp = val_a
    val_a = val_b
    val_b = temp
    return val_a, val_b
