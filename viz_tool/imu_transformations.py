# See https://docs.pupil-labs.com/alpha-lab/imu-transformations/
# for more details.

import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_imu_to_world(imu_coordinates, imu_quaternions):
    # This array contains a timeseries of transformation matrices,
    # as calculated from the IMU's timeseries of quaternions values.
    imu_to_world_matrices = R.from_quat(
        imu_quaternions,
        scalar_first=True,
    ).as_matrix()

    if np.ndim(imu_coordinates) == 1:
        return imu_to_world_matrices @ imu_coordinates
    else:
        return np.array(
            [
                imu_to_world @ imu_coord
                for imu_to_world, imu_coord in zip(
                    imu_to_world_matrices, imu_coordinates
                )
            ]
        )


def transform_scene_to_imu(
    coords_in_scene, translation_in_imu=np.array([0.0, -1.3, -6.62])
):
    imu_scene_rotation_diff = np.deg2rad(-90 - 12)
    scene_to_imu = np.array(
        [
            [1.0, 0.0, 0.0],
            [
                0.0,
                np.cos(imu_scene_rotation_diff),
                -np.sin(imu_scene_rotation_diff),
            ],
            [
                0.0,
                np.sin(imu_scene_rotation_diff),
                np.cos(imu_scene_rotation_diff),
            ],
        ]
    )

    coords_in_imu = scene_to_imu @ coords_in_scene.T

    coords_in_imu[0, :] += translation_in_imu[0]
    coords_in_imu[1, :] += translation_in_imu[1]
    coords_in_imu[2, :] += translation_in_imu[2]

    return coords_in_imu.T


def spherical_to_cartesian_scene(elevations, azimuths):
    """
    Convert Neon's spherical representation of 3D gaze to Cartesian coordinates.
    """

    elevations_rad = np.deg2rad(elevations)
    azimuths_rad = np.deg2rad(azimuths)

    # Elevation of 0 in Neon system corresponds to Y = 0, but
    # an elevation of 0 in traditional spherical coordinates would
    # correspond to Y = 1, so we convert elevation to the
    # more traditional format.
    elevations_rad += np.pi / 2

    # Azimuth of 0 in Neon system corresponds to X = 0, but
    # an azimuth of 0 in traditional spherical coordinates would
    # correspond to X = 1. Also, azimuth to the right in Neon is
    # more positive, whereas it is more negative in traditional
    # spherical coordiantes. So, we convert azimuth to the more
    # traditional format.
    azimuths_rad *= -1.0
    azimuths_rad += np.pi / 2

    return np.array(
        [
            np.sin(elevations_rad) * np.cos(azimuths_rad),
            np.cos(elevations_rad),
            np.sin(elevations_rad) * np.sin(azimuths_rad),
        ]
    ).T


def transform_scene_to_world(
    coords_in_scene, imu_quaternions, translation_in_imu=np.array([0.0, -1.3, -6.62])
):
    coords_in_imu = transform_scene_to_imu(coords_in_scene, translation_in_imu)
    return transform_imu_to_world(coords_in_imu, imu_quaternions)


def gaze_3d_to_world(gaze_elevation, gaze_azimuth, imu_quaternions):
    cart_gazes_in_scene = spherical_to_cartesian_scene(gaze_elevation, gaze_azimuth)
    return transform_scene_to_world(
        cart_gazes_in_scene, imu_quaternions, translation_in_imu=np.zeros(3)
    )


def imu_heading_in_world(imu_quaternions):
    heading_neutral_in_imu_coords = np.array([0.0, 1.0, 0.0])
    return transform_imu_to_world(heading_neutral_in_imu_coords, imu_quaternions)


def cartesian_to_spherical_world(world_points_3d):
    """
    Convert points in 3D Cartesian world coordinates to spherical coordinates.

    For elevation:
      - Neutral orientation = 0 (i.e., parallel with horizon)
      - Upwards is positive
      - Downwards is negative

    For azimuth:
      - Neutral orientation = 0 (i.e., aligned with magnetic North)
      - Leftwards is positive
      - Rightwards is negative
    """

    x = world_points_3d[:, 0]
    y = world_points_3d[:, 1]
    z = world_points_3d[:, 2]

    radii = np.sqrt(x**2 + y**2 + z**2)

    elevation = -(np.arccos(z / radii) - np.pi / 2)
    azimuth = np.arctan2(y, x) - np.pi / 2

    # Keep all azimuth values in the range of [-180, 180] to remain
    # consistent with the yaw orientation values provided by the IMU.
    azimuth[azimuth < -np.pi] += 2 * np.pi
    azimuth[azimuth > np.pi] -= 2 * np.pi

    elevation = np.rad2deg(elevation)
    azimuth = np.rad2deg(azimuth)

    return elevation, azimuth