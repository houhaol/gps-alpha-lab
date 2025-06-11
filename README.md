# pl-gps

This repository accompanies the Alpha Lab guide, [Use GPS with Neon to measure the wearer’s location, eye, and head movements](https://docs.pupil-labs.com/alpha-lab/gps/).

It contains the source code for an Android application that collects GPS data in a manner that complements Neon recordings.

It also contains Python code for a Visualization Tool that allows you to review combined eyetracking and GPS data. The Visualization Tool can accept data from any GPS device, so long as they have been put in a CSV file with the following format:

```
timestamp [ns],latitude,longitude
```

If you used your own GPS device, you will likely first need to post-hoc synchronize the data with your Neon recording.

## To make a GPS recording with the pl-gps app

- [Download](https://github.com/pupil-labs/pl_gps/releases) (or build) the pl-gps APK.
- Copy the APK to the Companion Device. A decent location is `Internal Storage/Documents`.
- Open the `Files` app on the Device. Then, find and install the APK.
  - You may see a pop-up and need to first give the `Files` app permission to install the app.
- Connect Neon and start up the Neon Companion app
- Do some figure-8 motions with the Neon _and_ the Companion Device, [as shown here](https://docs.pupil-labs.com/neon/data-collection/calibrating-the-imu/), so that they both have a good lock on magnetic north.
- Begin a Neon recording.
- Start up the pl-gps app, accept all permissions (if you have not done so yet), and tap the white button to start a GPS recording.
- Walk around and explore!
- If you walk past any landmarks of interest, simply click the `Send GPS Event` button in pl-gps.
  - You will need to connect the Companion Device to the hotspot of a second phone to enable this functionality.
  - If that hotspot is also connected to the Internet, then pl-gps will reverse geocode the Event on the fly, so that it shows up as an address name on Pupil Cloud. Otherwise, it will show up on Pupil Cloud as `gps_event`.
- When you are finished, first tap the red button in pl-gps, and then stop the Neon recording.
  - The pl-gps app will show a message with the name of the saved `gps … .csv` file. It will be in the `Documents/GPS` folder found in the `Files` app of the phone.
- Extract the saved GPS data to your computer either via a file syncing service, email, or via (USB cable using similar steps as when [exporting Neon recordings](https://docs.pupil-labs.com/neon/data-collection/transfer-recordings-via-usb/)).

## Visualization Tool

Now, you can load the Neon recording and GPS recorded data into the Visualization Tool, found in the `viz_tool` directory.

The Visualization Tool expects the `Timeseries CSV + Scene Video` download from Pupil Cloud.

Place Neon's scene video in a sub-directory of the `assets/` folder, named with the recording's Datetime UID (i.e., the name of the folder in `Timeseries Data` that contains your recording). For example, if your recording is in `Timeseries Data/2025-05-31_10-34-57-30558036`, then make an `assets/2025-05-31_10-34-57-30558036/` folder and put the Scene Video in there.

**Tip:** If you would like to see the gaze point in the video, then first run a [Video Renderer Visualization](https://docs.pupil-labs.com/neon/pupil-cloud/visualizations/video-renderer/) on Pupil Cloud for the recording and place that video in the appropriate sub-directory of the `assets/` folder.

You start the tool as follows:

```
python gps_viz_tool.py neon_timeseries_folder_filepath gps_csv_filepath
```

If you use [uv](https://docs.astral.sh/uv/), you can instead do:

```
uv run gps_viz_tool.py neon_timeseries_folder_filepath gps_csv_filepath
```

You can also pass an optional third parameter, `reverse_geocode`, to enable reverse geocoding of all events. Note that they will then be displayed with their address names.

Once started, you will see a web address listed in the terminal, typically http://127.0.0.1:8050/. Open this address in your web browser to view your data.

Briefly, the Visualization Tool shows three main panels:

- **Left:** A map with the wearer’s trajectory overlaid in blue. A black marker denotes the wearer's position. The IMU heading is shown as a black line, and the direction of gaze is shown as a red line. Positions corresponding to Events are shown as red markers.
- **Middle:** A video playback of the Neon recording.
- **Right:** A list of Events from the recording.

Clicking in the respective panel will jump to the corresponding points in the recording.

At the bottom, there are two dropdown selectors for `Start event` and `End event`. These can be used to limit the GPS trajectory to a subsection, making it easier to focus; for example, when wearers make several laps around a track.
