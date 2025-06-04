To make a GPS recording:

- Download the pl-gps APK
- Copy it to the Companion Device
- Open the `Files` app on the Device. Then, find and install the pl-gps APK
- Connect Neon, start up the Neon Companion app, and begin a Neon recording.
- Start up the pl-gps app, accept all permissions (if you have not done so yet), and tap the white button to start a GPS recording
- Walk around and explore!
- If you walk past any landmarks of interest, simply click the `Send GPS Event` button in pl-gps
    - You will need to connect the Companion Device to the hotspot of a second phone to enable this functionality.
    - If that hotspot is also connected to the Internet, then pl-gps will reverse geocode the Event on the fly, so that it shows up as an address name on Pupil Cloud. Otherwise, it will show up on Pupil Cloud as `gps_event` and will be later converted to an address name when loaded into the visualization tool.
- When you are finished, first tap the red button in pl-gps, and then stop the Neon recording.
    - The pl-gps app will show a message with the name of the saved `gps … .csv` file. It will be in the `Documents/GPS` folder found in the `Files` app of the phone.
- Extract the saved GPS data to your computer via USB cable, using similar steps as when [exporting Neon recordings](https://docs.pupil-labs.com/neon/data-collection/transfer-recordings-via-usb/).

Then, you can load the Neon recording and GPS recorded data into the Visualization Tool. The Visualization Tool expects the `Timeseries CSV + Scene Video` download from Pupil Cloud.

Briefly, the tool shows three main panels:

- **Left:** A map with the wearer’s trajectory overlaid in blue
- **Middle:** A video playback of the Neon recording
- **Right:** A list of Events from the recording
