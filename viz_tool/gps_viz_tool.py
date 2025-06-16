# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dash",
#     "dash-player",
#     "dash-leaflet",
#     "geopy",
#     "kaleido",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# ///
import argparse
import json
import math
import os
import sys

import dash
import dash_leaflet as dl
import dash_player as dp
import imu_transformations as imu_transformations
import numpy as np
import pandas as pd
from dash import ALL, Input, Output, dcc, html
from geopy.geocoders import Nominatim
from pie_arc import create_leaflet_pie_sector_coords
from scipy.interpolate import PchipInterpolator

# parse command line arguments for neon timeseries folder and gps csv file
parser = argparse.ArgumentParser(description="Neon GPS Visualization Tool")
parser.add_argument("neon_folder", help="Neon Timeseries CSV + Scene Video folder path")
parser.add_argument("gps_csv", help="GPS CSV file path")
parser.add_argument(
    "reverse_geocode", nargs="?", default=False, help="Reverse geocode events"
)

args = parser.parse_args()

neon_folder_path = args.neon_folder
gps_csv_path = args.gps_csv
reverse_geocode = args.reverse_geocode

if not os.path.isdir(neon_folder_path):
    print(f"Error: '{neon_folder_path}' is not a valid directory.", file=sys.stderr)
    sys.exit(1)


def open_and_populate_data():
    # load data
    with open(neon_folder_path + "/info.json") as f:
        info = json.load(f)

    # convert start_time to ISO format for compability with gps data + dash
    info["start_time_iso"] = pd.to_datetime(info["start_time"], unit="ns")
    info["start_timestamp"] = pd.to_datetime(info["start_time_iso"])

    # load the scene camera timestamps
    # to enable synced playback of GPS and Neon scene video
    world_df = pd.read_csv(neon_folder_path + "/world_timestamps.csv")
    world_df["timestamp_iso"] = pd.to_datetime(world_df["timestamp [ns]"], unit="ns")
    # add a column with row indices
    world_df["world_index"] = world_df.index
    world_df["rel timestamp [s]"] = (
        world_df["timestamp [ns]"] - world_df["timestamp [ns]"].min()
    )
    world_df["rel timestamp [s]"] = world_df["rel timestamp [s]"] / 1e9

    # load gaze and GPS data
    gps_df = pd.read_csv(gps_csv_path)

    # interpolate GPS data a bit to better match the scene camera timestamps
    lat_interp = PchipInterpolator(gps_df["timestamp [ns]"], gps_df["latitude"])
    lon_interp = PchipInterpolator(gps_df["timestamp [ns]"], gps_df["longitude"])
    # interp_tses = (
    #     np.arange(
    #         int(gps_df["timestamp [ns]"].min() * 1e-9),
    #         int(gps_df["timestamp [ns]"].max() * 1e-9),
    #         0.2,
    #     )
    #     * 1e9
    # )
    interp_tses = world_df["timestamp [ns]"].values
    gps_df = pd.DataFrame(
        {
            "timestamp [ns]": interp_tses,
            "latitude": lat_interp(interp_tses),
            "longitude": lon_interp(interp_tses),
        }
    )

    gaze = pd.read_csv(neon_folder_path + "/gaze.csv")

    # load imu data
    imu = pd.read_csv(neon_folder_path + "/imu.csv")
    imu["timestamp_iso"] = pd.to_datetime(imu["timestamp [ns]"], unit="ns")
    quaternions = np.array(
        [
            imu["quaternion w"],
            imu["quaternion x"],
            imu["quaternion y"],
            imu["quaternion z"],
        ]
    ).T

    # Resample the gaze azi/ele data to match the IMU timestamps
    gaze["timestamp_iso"] = pd.to_datetime(gaze["timestamp [ns]"], unit="ns")
    gaze_elevation_resampled = np.interp(
        imu["timestamp_iso"], gaze["timestamp_iso"], gaze["elevation [deg]"]
    )
    gaze_azimuth_resampled = np.interp(
        imu["timestamp_iso"], gaze["timestamp_iso"], gaze["azimuth [deg]"]
    )

    # use imu_transformations to convert gaze elevation and azimuth to world relative coordinates
    # see: https://docs.pupil-labs.com/alpha-lab/imu-transformations/
    cart_gazes_in_world = imu_transformations.gaze_3d_to_world(
        gaze_elevation_resampled, gaze_azimuth_resampled, quaternions
    )
    gazes_ele_world, gazes_azi_world = imu_transformations.cartesian_to_spherical_world(
        cart_gazes_in_world
    )

    # merge the resampled and transformed gaze data with the imu data
    # makes some steps later with visualization easier
    imu["gaze ele world [deg]"] = gazes_ele_world
    imu["gaze azi world [deg]"] = gazes_azi_world

    # load events
    events_df = pd.read_csv(neon_folder_path + "/events.csv")

    # Ensure all DataFrames have the same timestamp format
    imu["timestamp"] = pd.to_datetime(imu["timestamp_iso"])
    world_df["timestamp"] = pd.to_datetime(world_df["timestamp_iso"])
    gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp [ns]"])
    gaze["timestamp"] = pd.to_datetime(gaze["timestamp_iso"])
    events_df["timestamp_iso"] = pd.to_datetime(events_df["timestamp [ns]"], unit="ns")

    events_df["timestamp"] = pd.to_datetime(events_df["timestamp_iso"])

    # start merging the different dataframes into one
    # comprehensive dataframe
    gps_imu_df = pd.merge_asof(
        gps_df.sort_values("timestamp"),
        imu[
            ["timestamp", "yaw [deg]", "gaze ele world [deg]", "gaze azi world [deg]"]
        ].sort_values("timestamp"),
        on="timestamp",
    )
    gps_imu_df.set_index("timestamp", inplace=True)
    world_gps_imu_df = pd.merge_asof(
        gps_imu_df.sort_values("timestamp"),
        world_df[["timestamp", "world_index"]].sort_values("timestamp"),
        on="timestamp",
    )
    world_gps_imu_df.set_index("timestamp", inplace=True)
    world_gaze_gps_imu_df = pd.merge_asof(
        world_gps_imu_df.sort_values("timestamp"),
        gaze[["timestamp", "elevation [deg]", "azimuth [deg]"]].sort_values(
            "timestamp"
        ),
        on="timestamp",
    )
    world_gaze_gps_imu_df.set_index("timestamp", inplace=True)

    # make sure all dataframes have the same index
    # some actions are a bit easier later with just the gps dataframe
    # alone
    gps_df.set_index("timestamp", inplace=True)

    return (
        world_gaze_gps_imu_df,
        world_gps_imu_df,
        gps_imu_df,
        world_df,
        events_df,
        gps_df,
    )


def reverse_geocode_events(world_gaze_gps_imu_df, events_df):
    # reverse geocode the events
    geolocator = Nominatim(user_agent="my_reverse_geocoder")

    # having reverse geocoded events in a separate dataframe
    # with a copy of the gps coordinates makes the event selector
    # easier to implement and use
    event_gps_list = []
    for idx, row in events_df.iterrows():
        world_idx = world_gaze_gps_imu_df.index.get_indexer(
            [row["timestamp"]], method="nearest"
        )[0]
        world_row = world_gaze_gps_imu_df.iloc[world_idx]
        lat = world_row["latitude"]
        lon = world_row["longitude"]
        heading = world_row["yaw [deg]"]
        gaze_azi = world_row["gaze azi world [deg]"]

        if reverse_geocode:
            try:
                location = geolocator.reverse((lat, lon))

                event_gps_list.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "location": location,
                        "yaw [deg]": heading,
                        "gaze azi world [deg]": gaze_azi,
                        "timestamp": row["timestamp"],
                    }
                )
            except Exception:
                print("Could not reverse geocode event: ", row["name"])
        else:
            event_gps_list.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "location": row["name"],
                    "yaw [deg]": heading,
                    "gaze azi world [deg]": gaze_azi,
                    "timestamp": row["timestamp"],
                }
            )

    # transform it to a dataframe
    if reverse_geocode:
        geocoded_events_df = pd.DataFrame(
            {
                "lat": [event["lat"] for event in event_gps_list],
                "lon": [event["lon"] for event in event_gps_list],
                "location": [event["location"].address for event in event_gps_list],
                "size": [12 for event in event_gps_list],
            }
        )
    else:
        geocoded_events_df = pd.DataFrame(
            {
                "lat": [event["lat"] for event in event_gps_list],
                "lon": [event["lon"] for event in event_gps_list],
                "location": [event["location"] for event in event_gps_list],
                "size": [12 for event in event_gps_list],
            }
        )

    return geocoded_events_df, event_gps_list


def calculate_arrow_latlon_coords(lat, lon, heading, scale=0.0006):
    """
    Compute the end coordinates for an arrow based on a starting point (lat, lat),
    a heading (in degrees) and a scale factor.
    """
    theta = math.radians(heading)
    dlat = scale * math.sin(theta)
    dlon = scale * math.cos(theta)
    return lat + dlat, lon + dlon


def calculate_frustum_latlon_coords(lat, lon, heading, scale=0.0006):
    theta = math.radians(heading)
    lat_rad = math.radians(lat)
    angle_offset = math.radians(50)

    left_theta = theta + angle_offset
    right_theta = theta - angle_offset

    left_dlat = scale * math.sin(left_theta)
    right_dlat = scale * math.sin(right_theta)

    left_dlon = scale * math.cos(left_theta) / math.cos(lat_rad)
    right_dlon = scale * math.cos(right_theta) / math.cos(lat_rad)

    left_corner = (lat + left_dlat, lon + left_dlon)
    right_corner = (lat + right_dlat, lon + right_dlon)

    return left_corner, right_corner


def make_frustum_base(lats, lons):
    frustum_layer = {
        "source": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[lons[i], lats[i]] for i in range(len(lons))],
                    },
                }
            ],
        },
        "line": {
            "width": 0,
        },
        "type": "fill",
        "below": "traces",
        "color": "rgb(0, 20, 220, 0.2)",
        "name": "myTriangle",
    }

    return frustum_layer


def find_neon_video_path(neon_folder_path):
    datetime_uid = neon_folder_path.split("/")[1]
    for filename in os.listdir("./assets/" + datetime_uid):
        if filename.endswith(".mp4"):
            neon_scene_filename = filename

    if neon_scene_filename is None:
        print(
            "Error: No Neon scene video in the 'assets/`recording_id`' subdirectory. Please read the instructions.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        neon_scene_path = os.path.join("./assets/" + datetime_uid, neon_scene_filename)
        return neon_scene_path


# 1. Define the properties of the pie
maximum_radius = 0.001  # Radius in degrees
pie_start_angle = -50  # Start angle to create a 100-degree arc
pie_end_angle = 50  # End angle to create a 100-degree arc
number_of_gradient_layers = 40  # Increased for a smoother gradient
pie_color = "#007BFF"  # Bright blue


def create_base_map(world_gaze_gps_imu_df, world_df, geocoded_events_df):
    center_lat = world_gaze_gps_imu_df["latitude"].mean()
    center_lon = world_gaze_gps_imu_df["longitude"].mean()

    # Add the wearer pos and arrows to the map that corresponds to earliest scene camera frame
    target_timestamp = pd.Timedelta(seconds=0) + world_df["timestamp"].min()
    idx = world_gaze_gps_imu_df.index.get_indexer([target_timestamp], method="nearest")[
        0
    ]
    row = world_gaze_gps_imu_df.iloc[idx]

    initial_lat = row["latitude"]
    initial_lon = row["longitude"]
    # inital_heading = row["yaw [deg]"] + 90
    # initial_gaze_azi = row["gaze azi world [deg]"] + 90

    final_lat = world_gaze_gps_imu_df.iloc[len(world_gaze_gps_imu_df) - 1]["latitude"]
    final_lon = world_gaze_gps_imu_df.iloc[len(world_gaze_gps_imu_df) - 1]["longitude"]

    # add markers for all events
    event_markers = [
        dl.CircleMarker(
            center=[event[1].lat, event[1].lon],
            radius=4,
            color="red",
            fill=True,
            fillColor="red",
            opacity=1.0,
            fillOpacity=1.0,
            stroke=True,
            children=[dl.Tooltip(content=event[1].location)],
            id=f"event-marker-{idx + 1}",
        )
        for idx, event in enumerate(geocoded_events_df.iterrows())
    ]

    # Create concentric sectors from largest (most transparent) to smallest (most opaque)
    pie_arc = []
    pc = 0
    for i in range(number_of_gradient_layers, 0, -1):
        radius = (i / number_of_gradient_layers) * maximum_radius
        progress = (
            (number_of_gradient_layers - i) / (number_of_gradient_layers - 1)
            if number_of_gradient_layers > 1
            else 1
        )
        opacity = progress * 0.2

        # Get the coordinates for the current sector
        sector_coords = create_leaflet_pie_sector_coords(
            initial_lat, initial_lon, radius, pie_start_angle, pie_end_angle
        )

        # Each layer is a dash_leaflet Polygon component.
        # We set weight=0 to make the border invisible.
        pie_arc.append(
            dl.Polygon(
                positions=sector_coords,
                fillColor=pie_color,
                fillOpacity=opacity,
                stroke=False,  # An alternative way to ensure no border
                weight=0,
                id={"type": "pie-arc", "index": pc},
            )
        )
        pc += 1

    map = dl.Map(
        attributionControl=False,
        children=[
            dl.TileLayer(
                detectRetina=True,
            ),
            dl.Polyline(
                positions=world_gaze_gps_imu_df[["latitude", "longitude"]].values,
                color="blue",
                weight=1.5,
                id="wearer-trajectory",
            ),
            dl.CircleMarker(
                center=[initial_lat, initial_lon],
                radius=4,
                color="orange",
                fill=True,
                fillColor="orange",
                opacity=1.0,
                fillOpacity=1.0,
                stroke=True,
                children=[dl.Tooltip(content="GPS Start")],
                id="start-gps-point",
            ),
            dl.CircleMarker(
                center=[final_lat, final_lon],
                radius=4,
                color="orange",
                fill=True,
                fillColor="orange",
                opacity=1.0,
                fillOpacity=1.0,
                stroke=True,
                children=[dl.Tooltip(content="GPS End")],
                id="end-gps-point",
            ),
            dl.LayerGroup(event_markers, id="event-layer"),
            dl.LayerGroup(pie_arc, id="pie-arc-parent"),
            dl.CircleMarker(
                center=[initial_lat, initial_lon],
                radius=8,
                color="black",
                fill=True,
                fillColor="black",
                opacity=1.0,
                fillOpacity=1.0,
                stroke=True,
                children=[dl.Tooltip(content="Wearer")],
                id="wearer-marker",
            ),
            dl.Polyline(
                positions=[
                    [initial_lat, initial_lon],
                    [initial_lat, initial_lon],
                ],
                color="red",
                weight=4,
                id="gaze-arrow",
            ),
        ],
        center=[center_lat, center_lon],
        zoom=15,
        style={"height": "50vh"},
        id="map-graph",
    )

    return map


# load up all data, prepare fig, find neon scene video
(
    world_gaze_gps_imu_df,
    world_gps_imu_df,
    gps_imu_df,
    world_df,
    events_df,
    gps_df,
) = open_and_populate_data()

geocoded_events_df, event_gps_list = reverse_geocode_events(
    world_gaze_gps_imu_df, events_df
)

map = create_base_map(world_gaze_gps_imu_df, world_df, geocoded_events_df)
neon_scene_path = find_neon_video_path(neon_folder_path)

app_event_options = []
if reverse_geocode:
    app_event_options = [
        {"label": event["location"].address, "value": idx + 1}
        for idx, event in enumerate(event_gps_list)
    ]
else:
    app_event_options = [
        {"label": event["location"], "value": idx + 1}
        for idx, event in enumerate(event_gps_list)
    ]


app = dash.Dash(__name__, prevent_initial_callbacks=True)
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        map,
                        dcc.Interval(id="interval", interval=330, n_intervals=0),
                    ],
                    style={"flex": 1},
                ),
                html.Div(
                    [
                        dp.DashPlayer(
                            id="video-player",
                            url=neon_scene_path,
                            controls=True,
                            playing=False,
                            width="100%",
                            height="50vh",
                            intervalCurrentTime=40,
                            seekTo=0,
                        )
                    ],
                    style={"flex": 1},
                ),
                html.Div(
                    [
                        html.H4("Events"),
                        dcc.RadioItems(
                            id="gps-event-selector",
                            options=app_event_options,
                            value=None,
                            labelStyle={"display": "block"},
                        ),
                    ],
                    style={
                        "flex": 1,
                        "overflowY": "scroll",
                        "maxHeight": "50vh",
                        "border": "1px solid #ccc",
                    },
                ),
            ],
            style={"display": "flex"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        "Start event:",
                        dcc.Dropdown(
                            id="event-dropdown-1",
                            options=app_event_options,
                            value=1,
                        ),
                    ],
                    style={"flex": 1, "padding": "10px"},
                ),
                html.Div(
                    [
                        "End event:",
                        dcc.Dropdown(
                            id="event-dropdown-2",
                            options=app_event_options,
                            value=len(event_gps_list),
                        ),
                    ],
                    style={"flex": 1, "padding": "10px"},
                ),
            ],
            style={"flex": 1, "padding": "30px"},
        ),
    ]
)


global subset_df
subset_df = None

global trimmed
trimmed = False

global prev_selected_event
prev_selected_event = None

global trim_event1
trim_event1 = None

global trim_event2
trim_event2 = None


# define all the Dash callbacks that enable user interaction.
# they are called and managed by the Dash framework
@app.callback(
    Output("wearer-marker", "center", allow_duplicate=True),
    Output("gaze-arrow", "positions", allow_duplicate=True),
    Output({"type": "pie-arc", "index": ALL}, "positions", allow_duplicate=True),
    Input("video-player", "currentTime"),
)
def map_update_on_currentTime(currentTime):
    global subset_df
    global trimmed
    if currentTime is None:
        return dash.no_update

    target_timestamp = pd.Timedelta(seconds=currentTime) + world_df["timestamp"].min()

    idx = 0
    df_len = 0
    df_to_sample = []
    if trimmed:
        idx = subset_df.index.get_indexer([target_timestamp], method="nearest")[0]
        df_len = len(subset_df)
        df_to_sample = subset_df
    else:
        idx = gps_imu_df.index.get_indexer([target_timestamp], method="nearest")[0]
        df_len = len(gps_imu_df)
        df_to_sample = gps_imu_df

    if idx < df_len:
        row = df_to_sample.iloc[idx]
        new_lon = row["longitude"]
        new_lat = row["latitude"]
        heading = row["yaw [deg]"] + 90
        gaze_azi = row["gaze azi world [deg]"] + 90

        pie_heading = None
        if np.isnan(heading):
            pie_heading = 0
        else:
            pie_heading = heading

        pie_positions = []
        for i in range(number_of_gradient_layers, 0, -1):
            radius = (i / number_of_gradient_layers) * maximum_radius

            # Get the coordinates for the current sector
            sector_coords = create_leaflet_pie_sector_coords(
                new_lat,
                new_lon,
                radius,
                pie_start_angle + pie_heading,
                pie_end_angle + pie_heading,
            )
            pie_positions.append(sector_coords)

        if np.isnan(gaze_azi):
            new_x = new_lon
            new_y = new_lat
        else:
            new_x = new_lon + 0.0006 * np.cos(np.radians(gaze_azi))
            new_y = new_lat + 0.0006 * np.sin(np.radians(gaze_azi))

        return (
            [new_lat, new_lon],
            [[new_lat, new_lon], [new_y, new_x]],
            pie_positions,
        )

    return dash.no_update


@app.callback(
    Output("video-player", "seekTo", allow_duplicate=True),
    Input("event-dropdown-1", "value"),
    Input("event-dropdown-2", "value"),
)
def update_video_on_event_selection(start_event, end_event):
    global trim_event1
    global trim_event2
    if start_event is not None and end_event is not None:
        # Get the start event's timestamp and convert it to seconds.
        start_timestamp = (
            event_gps_list[start_event - 1]["timestamp"] - world_gps_imu_df.index.min()
        )
        start_timestamp = start_timestamp.total_seconds()
        trim_event1 = start_event

        # Get the end event's timestamp and convert it to seconds.
        end_timestamp = (
            event_gps_list[end_event - 1]["timestamp"] - world_gps_imu_df.index.min()
        )
        end_timestamp = end_timestamp.total_seconds()
        trim_event2 = end_event

        # Return the start timestamp to seek the video to that point.
        return start_timestamp

    return dash.no_update


@app.callback(
    Output("wearer-trajectory", "positions", allow_duplicate=True),
    Input("event-dropdown-1", "value"),
    Input("event-dropdown-2", "value"),
)
def update_map_on_event_selection(start_event, end_event):
    global subset_df
    global trimmed
    # global trim_event1
    # global trim_event2
    if start_event is not None and end_event is not None:
        start_timestamp = event_gps_list[start_event - 1]["timestamp"]
        end_timestamp = event_gps_list[end_event - 1]["timestamp"]

        # trim_event1 = start_event
        # trim_event2 = end_event

        subset_df = world_gaze_gps_imu_df[
            (world_gaze_gps_imu_df.index >= start_timestamp)
            & (world_gaze_gps_imu_df.index <= end_timestamp)
        ]

        trimmed = True

        return subset_df[["latitude", "longitude"]].values

    return dash.no_update


@app.callback(
    Output("wearer-marker", "center", allow_duplicate=True),
    Output("gaze-arrow", "positions", allow_duplicate=True),
    Output({"type": "pie-arc", "index": ALL}, "positions", allow_duplicate=True),
    Input("map-graph", "clickData"),
)
def update_map_on_click(clickData):
    global subset_df
    global trimmed
    if clickData:
        point = clickData["latlng"]
        clicked_lon = point["lng"]
        clicked_lat = point["lat"]

        dist = 0
        if trimmed:
            dist = np.sqrt(
                (subset_df["latitude"] - clicked_lat) ** 2
                + (subset_df["longitude"] - clicked_lon) ** 2
            )
        else:
            dist = np.sqrt(
                (gps_imu_df["latitude"] - clicked_lat) ** 2
                + (gps_imu_df["longitude"] - clicked_lon) ** 2
            )

        point_index = np.argmin(dist.values)

        closest_lat = []
        closest_lon = []
        heading = []
        gaze_azi = []
        if trimmed:
            closest_lat = subset_df.iloc[point_index]["latitude"]
            closest_lon = subset_df.iloc[point_index]["longitude"]
            heading = subset_df.iloc[point_index]["yaw [deg]"] + 90
            gaze_azi = subset_df.iloc[point_index]["gaze azi world [deg]"] + 90
        else:
            closest_lat = gps_imu_df.iloc[point_index]["latitude"]
            closest_lon = gps_imu_df.iloc[point_index]["longitude"]
            heading = gps_imu_df.iloc[point_index]["yaw [deg]"] + 90
            gaze_azi = gps_imu_df.iloc[point_index]["gaze azi world [deg]"] + 90

        pie_positions = []
        for i in range(number_of_gradient_layers, 0, -1):
            radius = (i / number_of_gradient_layers) * maximum_radius

            # Get the coordinates for the current sector
            sector_coords = create_leaflet_pie_sector_coords(
                closest_lat,
                closest_lon,
                radius,
                pie_start_angle + heading,
                pie_end_angle + heading,
            )
            pie_positions.append(sector_coords)

        new_x = closest_lon + 0.0006 * np.cos(np.radians(gaze_azi))
        new_y = closest_lat + 0.0006 * np.sin(np.radians(gaze_azi))

        return (
            [closest_lat, closest_lon],
            [[closest_lat, closest_lon], [new_y, new_x]],
            pie_positions,
        )
    else:
        return dash.no_update


# Callback to update the map position when the video is clicked.
@app.callback(
    Output("video-player", "seekTo", allow_duplicate=True),
    Input("gps-event-selector", "value"),
    Input("map-graph", "clickData"),
)
def seek_video(selected_gps_event, clickData):
    global prev_selected_event
    global subset_df
    global trimmed
    global trim_event1
    global trim_event2
    if selected_gps_event and selected_gps_event != prev_selected_event:
        # Reset the clickData to None when a new event is selected
        clickData = None
        prev_selected_event = selected_gps_event

    if clickData:
        point = clickData["latlng"]
        clicked_lon = point["lng"]
        clicked_lat = point["lat"]

        dist = 0
        if trimmed:
            dist = np.sqrt(
                (subset_df["latitude"] - clicked_lat) ** 2
                + (subset_df["longitude"] - clicked_lon) ** 2
            )
        else:
            dist = np.sqrt(
                (world_gps_imu_df["latitude"] - clicked_lat) ** 2
                + (world_gps_imu_df["longitude"] - clicked_lon) ** 2
            )

        point_index = np.argmin(dist.values)

        # Get the corresponding timestamp from the dataframe.
        timestamp = 0
        if trimmed:
            timestamp = subset_df.index[point_index] - world_gps_imu_df.index.min()
        else:
            timestamp = (
                world_gps_imu_df.index[point_index] - world_gps_imu_df.index.min()
            )

        # Convert the timestamp to seconds.
        timestamp = timestamp.total_seconds()
        return timestamp
    elif selected_gps_event:
        # Get the selected event's timestamp and convert it to seconds.
        # if selected_gps_event < trim_event1:
        # selected_gps_event = trim_event1
        # elif selected_gps_event > trim_event2:
        # selected_gps_event = trim_event2

        selected_event = event_gps_list[selected_gps_event - 1]
        timestamp = selected_event["timestamp"] - world_gps_imu_df.index.min()
        timestamp = timestamp.total_seconds()
        return timestamp

    return dash.no_update


if __name__ == "__main__":
    app.run(debug=True)
