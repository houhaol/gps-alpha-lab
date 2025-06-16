import numpy as np


def create_leaflet_pie_sector_coords(
    center_lat, center_lon, radius, start_angle, end_angle, num_segments=50
):
    """
    Creates the coordinate list for a Leaflet Polygon that represents a pie sector.
    Note: Leaflet expects coordinates as [latitude, longitude].

    Args:
        center_lat (float): Latitude of the center of the pie.
        center_lon (float): Longitude of the center of the pie.
        radius (float): The radius of the pie sector in degrees.
        start_angle (float): The starting angle of the sector in degrees.
        end_angle (float): The ending angle of the sector in degrees.
        num_segments (int): The number of line segments to use to approximate the arc.

    Returns:
        list: A list of [lat, lon] coordinates for a dash-leaflet Polygon.
    """
    # Create a list of angles for the arc points
    angles = np.linspace(start_angle, end_angle, num_segments)

    # Start the coordinate list at the center
    coords = [[center_lat, center_lon]]

    # Add points along the arc
    for angle in angles:
        # Convert radius from degrees to an approximate meter equivalent for more consistent sizing
        # This is a simplification; for true meter-based radius, a projection is needed.
        # But for visualization, this works well.
        x = center_lon + radius * np.cos(np.deg2rad(angle)) / np.cos(
            np.deg2rad(center_lat)
        )
        y = center_lat + radius * np.sin(np.deg2rad(angle))
        coords.append([y, x])

    # Close the polygon by returning to the center
    coords.append([center_lat, center_lon])

    return coords
