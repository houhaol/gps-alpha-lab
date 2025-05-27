import math
import json

import dash
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Input, Output, State, dcc, html
import dash_player as dp
from geopy.geocoders import Nominatim
import plotly.graph_objects as go

import imu_transformations as imu_transformations


with open("Timeseries Data + Scene Video 2/2025-03-18_15-34-47-db789c53/info.json") as f:
    data = json.load(f)

data["start_time_iso"] = pd.to_datetime(data["start_time"], unit="ns")
data["start_timestamp"] = pd.to_datetime(data["start_time_iso"])

# Sample data: GPS coordinates and corresponding IMU heading (in degrees)
# data = {
#     "lat": [37.77, 37.78, 37.79, 37.80],
#     "lon": [-122.42, -122.41, -122.40, -122.39],
#     "imu_heading": [0, 45, 90, 135],  # IMU heading values in degrees
# }
# df = pd.DataFrame(data)

gps = pd.read_csv("gps_1742309607770.csv")
gaze = pd.read_csv("Timeseries Data + Scene Video 2/2025-03-18_15-34-47-db789c53/gaze.csv")

imu = pd.read_csv("Timeseries Data + Scene Video 2/2025-03-18_15-34-47-db789c53/imu.csv")
imu["timestamp_iso"] = pd.to_datetime(imu["timestamp [ns]"], unit="ns")

quaternions = np.array([
    imu["quaternion w"],
    imu["quaternion x"],
    imu["quaternion y"],
    imu["quaternion z"],
]).T


# Resample the gaze data to match the IMU timestamps
gaze["timestamp_iso"] = pd.to_datetime(gaze["timestamp [ns]"], unit="ns")
gaze_elevation_resampled = np.interp(
    imu["timestamp_iso"], gaze["timestamp_iso"], gaze["elevation [deg]"]
)
gaze_azimuth_resampled = np.interp(imu["timestamp_iso"], gaze["timestamp_iso"], gaze["azimuth [deg]"])

cart_gazes_in_world = imu_transformations.gaze_3d_to_world(
    gaze_elevation_resampled, gaze_azimuth_resampled, quaternions
)

gazes_ele_world, gazes_azi_world = imu_transformations.cartesian_to_spherical_world(cart_gazes_in_world)
imu["gaze ele world [deg]"] = gazes_ele_world
imu["gaze azi world [deg]"] = gazes_azi_world


world = pd.read_csv("Timeseries Data + Scene Video 2/2025-03-18_15-34-47-db789c53/world_timestamps.csv")
world["timestamp_iso"] = pd.to_datetime(world["timestamp [ns]"], unit="ns")
# add a column with row indices
world["world_index"] = world.index
world["rel timestamp [s]"] = world["timestamp [ns]"] - world["timestamp [ns]"].min()
world["rel timestamp [s]"] = world["rel timestamp [s]"] / 1e9

# Ensure both DataFrames have the same timestamp format
imu["timestamp"] = pd.to_datetime(imu["timestamp_iso"])
world["timestamp"] = pd.to_datetime(world["timestamp_iso"])
gps["timestamp"] = pd.to_datetime(gps["timestamp"])
gaze["timestamp"] = pd.to_datetime(gaze["timestamp_iso"])

df = pd.merge_asof(
    gps.sort_values("timestamp"),
    imu[["timestamp", "yaw [deg]", "gaze ele world [deg]", "gaze azi world [deg]"]].sort_values("timestamp"),
    on="timestamp",
)
df.set_index("timestamp", inplace=True)
world_gps_df = pd.merge_asof(
    df.sort_values("timestamp"),
    world[["timestamp", "world_index"]].sort_values("timestamp"),
    on="timestamp",
)
world_gps_df.set_index("timestamp", inplace=True)
world_gaze_gps_df = pd.merge_asof(
    world_gps_df.sort_values("timestamp"),
    gaze[["timestamp", "elevation [deg]", "azimuth [deg]"]].sort_values("timestamp"),
    on="timestamp",
)
world_gaze_gps_df.set_index("timestamp", inplace=True)

gps.set_index("timestamp", inplace=True)
events = pd.read_csv("Timeseries Data + Scene Video 2/2025-03-18_15-34-47-db789c53/events.csv")
events["timestamp_iso"] = pd.to_datetime(events["timestamp [ns]"], unit="ns")
events["timestamp"] = pd.to_datetime(events["timestamp_iso"])


geolocator = Nominatim(user_agent="my_reverse_geocoder")

event_gps = []
for idx, row in events.iterrows():
    idx = world_gaze_gps_df.index.get_indexer([row["timestamp"]], method="nearest")[0]
    world_row = world_gaze_gps_df.iloc[idx]
    lat = world_row["latitude"]
    lon = world_row["longitude"]
    heading = world_row["yaw [deg]"]
    gaze_azi = world_row["gaze azi world [deg]"]

    location = geolocator.reverse((lat, lon))

    event_gps.append({
        "lat": lat,
        "lon": lon,
        "location": location,
        "yaw [deg]": heading,
        "gaze azi world [deg]": gaze_azi,
        "timestamp": row["timestamp"],
    })


def get_arrow_coords(lon, lat, heading, scale=0.0002):
    """
    Compute the end coordinates for an arrow based on a starting point (lon, lat),
    a heading (in degrees) and a scale factor.
    """
    theta = math.radians(heading)
    dlon = scale * math.cos(theta)
    dlat = scale * math.sin(theta)
    return lon + dlon, lat + dlat



# initial_time = df["timestamp"].min()
# subset = df[df["timestamp"] <= initial_time]
# Create an initial map figure using Plotly Express's line_map
fig = px.line_map(df, lat="latitude", lon="longitude", zoom=16, height=500)

target_timestamp = pd.Timedelta(seconds=0) + world["timestamp"].min()
idx = df.index.get_indexer([target_timestamp], method="nearest")[0]
row = df.iloc[idx]
print(idx, row["longitude"], row["latitude"])

clicked_lon = row["longitude"]
clicked_lat = row["latitude"]
heading = row["yaw [deg]"] + 90
gaze_azi = row["gaze azi world [deg]"] + 90

arrow_lon, arrow_lat = get_arrow_coords(clicked_lon, clicked_lat, heading)
arrow_df = pd.DataFrame(
    {"lat": [clicked_lat, arrow_lat], "lon": [clicked_lon, arrow_lon]}
)
# arrow_trace = px.scatter_map(
    # arrow_df, lat="lat", lon="lon", color_discrete_sequence=["black"]
# )
# for trace in arrow_trace.data:
# fig.add_trace(arrow_trace)

heading_line_trace = px.line_map(
    arrow_df, lat="lat", lon="lon", color_discrete_sequence=["black"]
)
for trace in heading_line_trace.data:
    fig.add_trace(trace)


gaze_arrow_lon, gaze_arrow_lat = get_arrow_coords(clicked_lon, clicked_lat, gaze_azi)
gaze_arrow_df = pd.DataFrame(
    {"lat": [clicked_lat, gaze_arrow_lat], "lon": [clicked_lon, gaze_arrow_lon]}
)
# gaze_arrow_trace = px.scatter_map(
    # gaze_arrow_df, lat="lat", lon="lon", color_discrete_sequence=["red"]
# )
# for trace in gaze_arrow_trace.data:
    # fig.add_trace(trace)

gaze_line_trace = px.line_map(
    gaze_arrow_df, lat="lat", lon="lon", color_discrete_sequence=["red"]
)
for trace in gaze_line_trace.data:
    fig.add_trace(trace)


events_df = pd.DataFrame(
    {
        "lat": [event["lat"] for event in event_gps],
        "lon": [event["lon"] for event in event_gps],
        "location": [event["location"].address for event in event_gps],
        "size": [20 for event in event_gps],
    }
)
# events_markers = px.scatter_map(
    # events_df, lat="lat", lon="lon", size="size"
# )
# for trace in events_markers.data:
    # fig.add_trace(trace)

events_markers = go.Scattermap(
    lat=events_df["lat"],
    lon=events_df["lon"],
    mode="markers",
    marker=go.scattermap.Marker(
        size=20,
        color="red",
        opacity=0.9,
    ),
)
fig.add_trace(events_markers)

# Use an OpenStreetMap basemap (no token needed)
fig.update_layout(
    map_style="open-street-map",
    clickmode="event+select",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    showlegend=False,
)

app = dash.Dash(__name__, prevent_initial_callbacks=True)
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Graph(id="map-graph", figure=fig), # Interval component to update every 1 second (1000 milliseconds)
            dcc.Interval(id="interval", interval=330, n_intervals=0)
        ], style={'flex': 1, 'padding': '10px'}),
        html.Div([
            dp.DashPlayer(
                id="video-player",
                url="/assets/8b646084_0.0-1130.591.mp4",
                controls=True,
                playing=False,
                width="100%",
                height=500,
                intervalCurrentTime=330,
                seekTo=0,
            )
        ], style={'flex': 1, 'padding': '10px'}),
        html.Div([
            html.H4("Events"),
            dcc.RadioItems(
                id='gps-event-selector',
                options=[
                    {'label': event["location"].address, 'value': idx + 1}
                    for idx, event in enumerate(event_gps)
                ],
                value=None,
                labelStyle={'display': 'block'},)
        ], style={'flex': 1, 'padding': '10px', 'overflowY': 'scroll', 'maxHeight': '500px', 'border': '1px solid #ccc'}),
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            "Start event:",
            dcc.Dropdown(
                id="event-dropdown-1",
                options=[
                    {"label": event["location"].address, "value": idx}
                    for idx, event in enumerate(event_gps)
                ],
                value=0
            ),
        ], style={'flex': 1, 'padding': '10px'}),
        html.Div([
            "End event:",
            dcc.Dropdown(
                id="event-dropdown-2",
                options=[
                    {"label": event["location"].address, "value": idx}
                    for idx, event in enumerate(event_gps)
                ],
                value=len(event_gps) - 1
            ),
        ], style={'flex': 1, 'padding': '10px'}),
    ], style={'flex': 1, 'padding': '30px'}),
])

@app.callback(
    Output("map-graph", "figure", allow_duplicate=True),
    Input("video-player", "currentTime"),
    State("map-graph", "figure"),  # Capture the current map state
)
def map_currentTime(currentTime, current_fig, allow_duplicate=True):# Keep the same zoom and center
    # zoom = current_fig["layout"]["map"].get("zoom", 12)
    # center = current_fig["layout"]["map"].get(
        # "center", {"lat": 37.78, "lon": -122.41}
    # )

    # new_fig = go.Figure(fig)
    # new_fig = px.line_map(df, lat="latitude", lon="longitude", zoom=zoom, height=500)
    # new_fig.update_layout(
        # map_style="open-street-map",
        # clickmode="event+select",
        # map_center=center,
        # map_zoom=zoom,
    # )

    target_timestamp = pd.Timedelta(seconds=currentTime) + world["timestamp"].min()
    idx = df.index.get_indexer([target_timestamp], method="nearest")[0]

    # if row is not None:
    if idx < len(df):
        row = df.iloc[idx]
        print(idx, row["longitude"], row["latitude"])
        clicked_lon = row["longitude"]
        clicked_lat = row["latitude"]
        heading = row["yaw [deg]"] + 90
        gaze_azi = row["gaze azi world [deg]"] + 90

        arrow_lon, arrow_lat = get_arrow_coords(clicked_lon, clicked_lat, heading)

        # Compute new arrow coordinates
        new_x = clicked_lon + 0.0002 * np.cos(np.radians(heading))
        new_y = clicked_lat + 0.0002 * np.sin(np.radians(heading))

        print(new_x, new_y)

        # Modify only the heading trace (trace index 1)
        current_fig["data"][1]["lat"] = [clicked_lat, new_y]
        current_fig["data"][1]["lon"] = [clicked_lon, new_x]


        # Compute new arrow coordinates
        new_x = clicked_lon + 0.0002 * np.cos(np.radians(gaze_azi))
        new_y = clicked_lat + 0.0002 * np.sin(np.radians(gaze_azi))

        print(new_x, new_y)

        # Modify only the heading trace (trace index 1)
        current_fig["data"][2]["lat"] = [clicked_lat, new_y]
        current_fig["data"][2]["lon"] = [clicked_lon, new_x]

        # arrow_df = pd.DataFrame(
            # {"lat": [clicked_lat, arrow_lat], "lon": [clicked_lon, arrow_lon]}
        # )
        # arrow_trace = px.scatter_map(
            # arrow_df, lat="lat", lon="lon", color_discrete_sequence=["red"]
        # )
        # for trace in arrow_trace.data:
            # new_fig.add_trace(trace)

        # line_trace = px.line_map(
            # arrow_df, lat="lat", lon="lon", color_discrete_sequence=["red"]
        # )
        # for trace in line_trace.data:
        # new_fig.add_trace(trace)

        
        # gaze_arrow_lon, gaze_arrow_lat = get_arrow_coords(clicked_lon, clicked_lat, gaze_azi)
        # gaze_arrow_df = pd.DataFrame(
        #     {"lat": [clicked_lat, gaze_arrow_lat], "lon": [clicked_lon, gaze_arrow_lon]}
        # )
        # gaze_arrow_trace = px.scatter_map(
        #     gaze_arrow_df, lat="lat", lon="lon", color_discrete_sequence=["black"]
        # )
        # for trace in gaze_arrow_trace.data:
        #     new_fig.add_trace(trace)

        # line_trace = px.line_map(
        #     gaze_arrow_df, lat="lat", lon="lon", color_discrete_sequence=["black"]
        # )
        # for trace in line_trace.data:
        #     new_fig.add_trace(trace)

    return current_fig

global prev_selected_event
prev_selected_event = None
@app.callback(
    Output("map-graph", "figure", allow_duplicate=True),
    Input("map-graph", "clickData"),
    Input("gps-event-selector", "value"),
    State("map-graph", "figure"),  # Capture the current map state
)
def update_arrow(clickData, selected_gps_event, current_fig, allow_duplicate=True):# Calculate current playback time
    # Keep the same zoom and center
    # zoom = current_fig["layout"]["map"].get("zoom", 12)
    # center = current_fig["layout"]["map"].get(
        # "center", {"lat": 37.78, "lon": -122.41}
    # )

    # new_fig = px.line_map(df, lat="latitude", lon="longitude", zoom=zoom, height=500)
    # new_fig.update_layout(
        # map_style="open-street-map",
        # clickmode="event+select",
        # map_center=center,
        # map_zoom=zoom,
    # )

    global prev_selected_event
    if selected_gps_event and selected_gps_event != prev_selected_event:
        # Reset the clickData to None when a new event is selected
        clickData = None
        prev_selected_event = selected_gps_event
        print("selected event", selected_gps_event)
        print("prev event", prev_selected_event)

    if clickData:
        point = clickData["points"][0]
        clicked_lon = point["lon"]
        clicked_lat = point["lat"]
        point_index = point.get("pointIndex")
        if point_index is not None:
            heading = df.iloc[point_index]["yaw [deg]"] + 90
            gaze_azi = df.iloc[point_index]["gaze azi world [deg]"] + 90
            arrow_lon, arrow_lat = get_arrow_coords(clicked_lon, clicked_lat, heading)

            # Compute new arrow coordinates
            new_x = clicked_lon + 0.0002 * np.cos(np.radians(heading))
            new_y = clicked_lat + 0.0002 * np.sin(np.radians(heading))

            # Modify only the heading trace (trace index 1)
            current_fig["data"][1]["lat"] = [clicked_lat, new_y]
            current_fig["data"][1]["lon"] = [clicked_lon, new_x]


            # Compute new arrow coordinates
            new_x = clicked_lon + 0.0002 * np.cos(np.radians(gaze_azi))
            new_y = clicked_lat + 0.0002 * np.sin(np.radians(gaze_azi))

            print(new_x, new_y)

            # Modify only the heading trace (trace index 1)
            current_fig["data"][2]["lat"] = [clicked_lat, new_y]
            current_fig["data"][2]["lon"] = [clicked_lon, new_x]


            # arrow_df = pd.DataFrame(
            #     {"lat": [clicked_lat, arrow_lat], "lon": [clicked_lon, arrow_lon]}
            # )
            # arrow_trace = px.scatter_map(
            #     arrow_df, lat="lat", lon="lon", color_discrete_sequence=["red"]
            # )
            # for trace in arrow_trace.data:
            #     new_fig.add_trace(trace)

            # line_trace = px.line_map(
            #     arrow_df, lat="lat", lon="lon", color_discrete_sequence=["red"]
            # )
            # for trace in line_trace.data:
            #     new_fig.add_trace(trace)

            
            # gaze_arrow_lon, gaze_arrow_lat = get_arrow_coords(clicked_lon, clicked_lat, gaze_azi)
            # gaze_arrow_df = pd.DataFrame(
            #     {"lat": [clicked_lat, gaze_arrow_lat], "lon": [clicked_lon, gaze_arrow_lon]}
            # )
            # gaze_arrow_trace = px.scatter_map(
            #     gaze_arrow_df, lat="lat", lon="lon", color_discrete_sequence=["black"]
            # )
            # for trace in gaze_arrow_trace.data:
            #     new_fig.add_trace(trace)

            # line_trace = px.line_map(
            #     gaze_arrow_df, lat="lat", lon="lon", color_discrete_sequence=["black"]
            # )
            # for trace in line_trace.data:
            #     new_fig.add_trace(trace)
    elif selected_gps_event:
        row = event_gps[selected_gps_event - 1]
        print("selected event", selected_gps_event, row["lon"], row["lat"])

        clicked_lon = row["lon"]
        clicked_lat = row["lat"]
        heading = row["yaw [deg]"] + 90
        gaze_azi = row["gaze azi world [deg]"] + 90

        arrow_lon, arrow_lat = get_arrow_coords(clicked_lon, clicked_lat, heading)

        # Compute new arrow coordinates
        new_x = clicked_lon + 0.0002 * np.cos(np.radians(heading))
        new_y = clicked_lat + 0.0002 * np.sin(np.radians(heading))

        # Modify only the heading trace (trace index 1)
        current_fig["data"][1]["lat"] = [clicked_lat, new_y]
        current_fig["data"][1]["lon"] = [clicked_lon, new_x]


        # Compute new arrow coordinates
        new_x = clicked_lon + 0.0002 * np.cos(np.radians(gaze_azi))
        new_y = clicked_lat + 0.0002 * np.sin(np.radians(gaze_azi))

        print(new_x, new_y)

        # Modify only the heading trace (trace index 1)
        current_fig["data"][2]["lat"] = [clicked_lat, new_y]
        current_fig["data"][2]["lon"] = [clicked_lon, new_x]

    return current_fig


# Callback to update the video playback position when the map is clicked.
@app.callback(
    Output("video-player", "seekTo"),
    Input("gps-event-selector", "value"),
    Input("map-graph", "clickData")
)
def seek_video(selected_gps_event, clickData):
    global prev_selected_event
    if selected_gps_event and selected_gps_event != prev_selected_event:
        # Reset the clickData to None when a new event is selected
        clickData = None
        prev_selected_event = selected_gps_event
        print("selected event", selected_gps_event)
        print("prev event", prev_selected_event)

    if clickData:
        # Extract the point index from the clickData; adjust depending on your clickData structure.
        point_index = clickData["points"][0].get("pointIndex")
        if point_index is not None:
            # Get the corresponding timestamp from the dataframe.
            timestamp = world_gps_df.index[point_index] - world_gps_df.index.min()
            # Convert the timestamp to seconds.
            timestamp = timestamp.total_seconds()
            return timestamp
    elif selected_gps_event:
        # Get the selected event's timestamp and convert it to seconds.
        selected_event = event_gps[selected_gps_event - 1]
        timestamp = selected_event["timestamp"] - world_gps_df.index.min()
        timestamp = timestamp.total_seconds()
        return timestamp

    return dash.no_update


if __name__ == "__main__":
    app.run(debug=True)