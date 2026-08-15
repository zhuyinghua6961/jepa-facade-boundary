import numpy as np

from boundary_sweep.geometry import (
    camera_to_world,
    intrinsics_from_fov,
    pixel_depth_to_camera_point,
    pixel_to_camera_ray,
    ray_plane_intersection,
    surface_coordinate_to_world_point,
    world_point_to_surface_coordinate,
    world_to_camera,
    world_to_pixel,
)


class _V:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z


class _R:
    def __init__(self, pitch=0, yaw=0, roll=0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class _T:
    def __init__(self, location=None, rotation=None):
        self.location = location or _V()
        self.rotation = rotation or _R()


def test_intrinsics_and_center_projection():
    K = intrinsics_from_fov(640, 480, 90)
    assert np.allclose(K[0, 0], 320.0)
    p = world_to_pixel([10, 0, 0], _T(), K)
    assert np.allclose(p[:2], [319.5, 239.5])


def test_cv_ue_world_roundtrip():
    T = _T(_V(3, 4, 5), _R(yaw=25, pitch=-3, roll=4))
    point = np.array([1.2, -0.8, 8.0])
    assert np.allclose(camera_to_world(world_to_camera(point, T), T), point, atol=1e-9)


def test_depth_modes_are_distinct_and_ray_unit():
    K = intrinsics_from_fov(640, 480, 90)
    pixel = [479.5, 239.5]
    z_point = pixel_depth_to_camera_point(pixel, 10.0, K, "z-depth")
    range_point = pixel_depth_to_camera_point(pixel, 10.0, K, "ray-range")
    assert np.isclose(z_point[2], 10.0)
    assert np.isclose(np.linalg.norm(range_point), 10.0)
    assert not np.allclose(z_point, range_point)
    assert np.isclose(np.linalg.norm(pixel_to_camera_ray(pixel, K)), 1.0)


def test_ray_plane_intersection():
    hit = ray_plane_intersection([0, 0, 0], [0, 0, 1], [0, 0, 5], [0, 0, 1])
    assert np.allclose(hit, [0, 0, 5])
    assert ray_plane_intersection([0, 0, 0], [1, 0, 0], [0, 0, 5], [0, 0, 1]) is None


def test_surface_coordinates_roundtrip():
    origin = np.array([2, 3, 4.])
    h, v = np.array([0, 1, 0.]), np.array([0, 0, 1.])
    p = surface_coordinate_to_world_point([1.5, 2.0], origin, h, v)
    assert np.allclose(world_point_to_surface_coordinate(p, origin, h, v), [1.5, 2.0])

