# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dash",
#     "dash-player",
#     "geopy",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
# ]
# ///
import argparse
import json
import math
import os
import sys

import dash
import dash_player as dp
import imu_transformations as imu_transformations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from geopy.geocoders import Nominatim
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


def calculate_arrow_latlon_coords(lat, lon, heading, scale=0.0003):
    """
    Compute the end coordinates for an arrow based on a starting point (lat, lat),
    a heading (in degrees) and a scale factor.
    """
    theta = math.radians(heading)
    dlat = scale * math.sin(theta)
    dlon = scale * math.cos(theta)
    return lat + dlat, lon + dlon


# create base map figure
def create_base_map_figure(world_gaze_gps_imu_df, world_df, geocoded_events_df):
    # initial_time = df["timestamp"].min()
    # subset = df[df["timestamp"] <= initial_time]

    # Create an initial map figure using Plotly Express's line_map
    fig = px.line_map(
        world_gaze_gps_imu_df, lat="latitude", lon="longitude", zoom=16, height=500
    )

    # Use an OpenStreetMap basemap (no token needed)
    fig.update_layout(
        map_style="open-street-map",
        clickmode="event+select",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
        uirevision="constant",
    )

    # Add the wearer pos and arrows to the map that corresponds to earliest scene camera frame
    target_timestamp = pd.Timedelta(seconds=0) + world_df["timestamp"].min()
    idx = world_gaze_gps_imu_df.index.get_indexer([target_timestamp], method="nearest")[
        0
    ]
    row = world_gaze_gps_imu_df.iloc[idx]

    initial_lat = row["latitude"]
    initial_lon = row["longitude"]
    inital_heading = row["yaw [deg]"] + 90
    initial_gaze_azi = row["gaze azi world [deg]"] + 90

    # add arrow for imu heading
    heading_arrow_lon, heading_arrow_lat = calculate_arrow_latlon_coords(
        initial_lat, initial_lon, inital_heading
    )
    heading_arrow_df = pd.DataFrame(
        {
            "lat": [initial_lat, heading_arrow_lat],
            "lon": [initial_lon, heading_arrow_lon],
        }
    )
    heading_line_trace = px.line_map(
        heading_arrow_df, lat="lat", lon="lon", color_discrete_sequence=["black"]
    )
    for trace in heading_line_trace.data:
        fig.add_trace(trace)

    # add arrow for gaze direction
    gaze_arrow_lon, gaze_arrow_lat = calculate_arrow_latlon_coords(
        initial_lat, initial_lon, initial_gaze_azi
    )
    gaze_arrow_df = pd.DataFrame(
        {"lat": [initial_lat, gaze_arrow_lat], "lon": [initial_lon, gaze_arrow_lon]}
    )
    gaze_line_trace = px.line_map(
        gaze_arrow_df, lat="lat", lon="lon", color_discrete_sequence=["red"]
    )
    for trace in gaze_line_trace.data:
        fig.add_trace(trace)

    # add a marker for the wearer
    wearer_marker = go.Scattermap(
        fill="toself",
        fillcolor="black",
        lat=[0],
        lon=[0],
        mode="markers",
        marker=dict(size=18, color="black", opacity=1.0),
        opacity=1.0,
    )
    fig.add_trace(wearer_marker)

    # add markers for all events
    events_markers = go.Scattermap(
        lat=geocoded_events_df["lat"],
        lon=geocoded_events_df["lon"],
        mode="markers",
        marker=go.scattermap.Marker(
            size=12,
            color="red",
            opacity=0.9,
        ),
    )

    fig.add_trace(events_markers)

    return fig


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

fig = create_base_map_figure(world_gaze_gps_imu_df, world_df, geocoded_events_df)
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

# create the Dash app
# and define the layout
app = dash.Dash(__name__, prevent_initial_callbacks=True)
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="map-graph", figure=fig
                        ),  # Interval component to update every 1 second (1000 milliseconds)
                        dcc.Interval(id="interval", interval=330, n_intervals=0),
                    ],
                    style={"flex": 1, "padding": "10px"},
                ),
                html.Div(
                    [
                        dp.DashPlayer(
                            id="video-player",
                            url=neon_scene_path,
                            controls=True,
                            playing=False,
                            width="100%",
                            height=500,
                            intervalCurrentTime=330,
                            seekTo=0,
                        )
                    ],
                    style={"flex": 1, "padding": "10px"},
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
                        "padding": "10px",
                        "overflowY": "scroll",
                        "maxHeight": "500px",
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


# define all the Dash callbacks that enable user interaction.
# they are called and managed by the Dash framework
@app.callback(
    Output("map-graph", "figure", allow_duplicate=True),
    Input("video-player", "currentTime"),
    State("map-graph", "figure"),
)
def map_update_on_currentTime(currentTime, current_fig):
    target_timestamp = pd.Timedelta(seconds=currentTime) + world_df["timestamp"].min()
    idx = gps_imu_df.index.get_indexer([target_timestamp], method="nearest")[0]

    # if row is not None:
    if idx < len(gps_imu_df):
        row = gps_imu_df.iloc[idx]
        new_lon = row["longitude"]
        new_lat = row["latitude"]
        heading = row["yaw [deg]"] + 90
        gaze_azi = row["gaze azi world [deg]"] + 90

        # Compute new arrow coordinates
        new_x = new_lon + 0.0003 * np.cos(np.radians(heading))
        new_y = new_lat + 0.0003 * np.sin(np.radians(heading))

        # Modify the heading trace (trace index 1)
        current_fig["data"][1]["lat"] = [new_lat, new_y]
        current_fig["data"][1]["lon"] = [new_lon, new_x]

        # Compute new arrow coordinates
        new_x = new_lon + 0.0003 * np.cos(np.radians(gaze_azi))
        new_y = new_lat + 0.0003 * np.sin(np.radians(gaze_azi))

        # Modify the gaze trace (trace index 2)
        current_fig["data"][2]["lat"] = [new_lat, new_y]
        current_fig["data"][2]["lon"] = [new_lon, new_x]

        # Modify wearer position (trace index 3)
        current_fig["data"][3]["lat"] = [new_lat]
        current_fig["data"][3]["lon"] = [new_lon]

    return current_fig


@app.callback(
    Output("video-player", "seekTo", allow_duplicate=True),
    Input("event-dropdown-1", "value"),
    Input("event-dropdown-2", "value"),
)
def update_video_on_event_selection(start_event, end_event):
    if start_event is not None and end_event is not None:
        # Get the start event's timestamp and convert it to seconds.
        start_timestamp = (
            event_gps_list[start_event - 1]["timestamp"] - world_gps_imu_df.index.min()
        )
        start_timestamp = start_timestamp.total_seconds()

        # Get the end event's timestamp and convert it to seconds.
        end_timestamp = (
            event_gps_list[end_event - 1]["timestamp"] - world_gps_imu_df.index.min()
        )
        end_timestamp = end_timestamp.total_seconds()

        # Return the start timestamp to seek the video to that point.
        return start_timestamp

    return dash.no_update


@app.callback(
    Output("map-graph", "figure", allow_duplicate=True),
    Input("event-dropdown-1", "value"),
    Input("event-dropdown-2", "value"),
    State("map-graph", "figure"),
)
def update_map_on_event_selection(start_event, end_event, current_fig):
    if start_event is not None and end_event is not None:
        start_timestamp = event_gps_list[start_event - 1]["timestamp"]
        end_timestamp = event_gps_list[end_event - 1]["timestamp"]

        subset_df = world_gaze_gps_imu_df[
            (world_gaze_gps_imu_df.index >= start_timestamp)
            & (world_gaze_gps_imu_df.index <= end_timestamp)
        ]

        current_fig["data"][0]["lat"] = subset_df["latitude"].tolist()
        current_fig["data"][0]["lon"] = subset_df["longitude"].tolist()

        return current_fig

    return dash.no_update


@app.callback(
    Output("map-graph", "figure", allow_duplicate=True),
    Input("map-graph", "clickData"),
    State("map-graph", "figure"),
)
def update_map_on_click(clickData, current_fig):
    if clickData:
        point = clickData["points"][0]
        clicked_lon = point["lon"]
        clicked_lat = point["lat"]
        point_index = point.get("pointIndex")
        if point_index is not None:
            heading = gps_imu_df.iloc[point_index]["yaw [deg]"] + 90
            gaze_azi = gps_imu_df.iloc[point_index]["gaze azi world [deg]"] + 90

            clickData = None

            # Create an initial map figure using Plotly Express's line_map
            fig = px.line_map(
                world_gaze_gps_imu_df,
                lat="latitude",
                lon="longitude",
                zoom=16,
                height=500,
            )

            # Use an OpenStreetMap basemap (no token needed)
            fig.update_layout(
                map_style="open-street-map",
                clickmode="event+select",
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                showlegend=False,
                uirevision="constant",
            )

            heading_arrow_lon, heading_arrow_lat = calculate_arrow_latlon_coords(
                clicked_lat, clicked_lon, heading
            )
            heading_arrow_df = pd.DataFrame(
                {
                    "lat": [clicked_lat, heading_arrow_lat],
                    "lon": [clicked_lon, heading_arrow_lon],
                }
            )
            heading_line_trace = px.line_map(
                heading_arrow_df,
                lat="lat",
                lon="lon",
                color_discrete_sequence=["black"],
            )
            for trace in heading_line_trace.data:
                fig.add_trace(trace)

            # add arrow for gaze direction
            gaze_arrow_lon, gaze_arrow_lat = calculate_arrow_latlon_coords(
                clicked_lat, clicked_lon, gaze_azi
            )
            gaze_arrow_df = pd.DataFrame(
                {
                    "lat": [clicked_lat, gaze_arrow_lat],
                    "lon": [clicked_lon, gaze_arrow_lon],
                }
            )
            gaze_line_trace = px.line_map(
                gaze_arrow_df, lat="lat", lon="lon", color_discrete_sequence=["red"]
            )
            for trace in gaze_line_trace.data:
                fig.add_trace(trace)

            # add a marker for the wearer
            wearer_marker = go.Scattermap(
                fill="toself",
                fillcolor="black",
                lat=[0],
                lon=[0],
                mode="markers",
                marker=dict(size=18, color="black", opacity=1.0),
                opacity=1.0,  # <--- Force entire trace to be fully opaque
            )
            fig.add_trace(wearer_marker)

            # add markers for all events
            events_markers = go.Scattermap(
                lat=geocoded_events_df["lat"],
                lon=geocoded_events_df["lon"],
                mode="markers",
                marker=go.scattermap.Marker(
                    size=12,
                    color="red",
                    opacity=0.9,
                ),
            )
            fig.add_trace(events_markers)

            return fig

    return current_fig


global prev_selected_event
prev_selected_event = None


# Callback to update the map position when the video is clicked.
@app.callback(
    Output("video-player", "seekTo", allow_duplicate=True),
    Input("gps-event-selector", "value"),
    Input("map-graph", "clickData"),
)
def seek_video(selected_gps_event, clickData):
    global prev_selected_event
    if selected_gps_event and selected_gps_event != prev_selected_event:
        # Reset the clickData to None when a new event is selected
        clickData = None
        prev_selected_event = selected_gps_event

    if clickData:
        # Extract the point index from the clickData
        point_index = clickData["points"][0].get("pointIndex")
        if point_index is not None:
            # Get the corresponding timestamp from the dataframe.
            timestamp = (
                world_gps_imu_df.index[point_index] - world_gps_imu_df.index.min()
            )
            # Convert the timestamp to seconds.
            timestamp = timestamp.total_seconds()
            return timestamp
    elif selected_gps_event:
        # Get the selected event's timestamp and convert it to seconds.
        selected_event = event_gps_list[selected_gps_event - 1]
        timestamp = selected_event["timestamp"] - world_gps_imu_df.index.min()
        timestamp = timestamp.total_seconds()
        return timestamp

    return dash.no_update


if __name__ == "__main__":
    app.run(debug=True)
