
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
# Re-encode with ffmpeg for browser compatibility
import subprocess
import shutil


# Paths
DATA_DIR = '/mnt/d/BEFIT/ProcessedData/Study/BF002/web_vis/'
VIDEO_PATH = '/home/houhao/befit/gps-alpha-lab/viz_tool/assets/BF002/world.mp4'
FIXATIONS_PATH = os.path.join(DATA_DIR, 'fixations.csv')
TIMESTAMPS_PATH = os.path.join(DATA_DIR, 'world_timestamps_unix.npy')
OUTPUT_PATH = '/home/houhao/befit/gps-alpha-lab/viz_tool/assets/BF002/world_annotated.mp4'


# Load fixations with correct columns
fix_df = pd.read_csv(FIXATIONS_PATH)
# Columns: 'fixation id', 'start timestamp [ns]', 'end timestamp [ns]', 'duration [ms]', 'fixation x [px]', 'fixation y [px]'
# We'll use 'start timestamp [ns]' and 'fixation x [px]', 'fixation y [px]'
fix_df = fix_df.rename(columns={
    'start timestamp [ns]': 'start_ts_ns',
    'end timestamp [ns]': 'end_ts_ns',
    'fixation x [px]': 'x',
    'fixation y [px]': 'y'
})
fix_df['start_ts_s'] = fix_df['start_ts_ns'] / 1e9  # convert ns to seconds


# Load frame timestamps
frame_timestamps = np.load(TIMESTAMPS_PATH)
num_frames = len(frame_timestamps)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))


# For fast lookup, sort fixations by start_ts_s
fix_df = fix_df.sort_values('start_ts_s')
fix_idx = 0
num_fix = len(fix_df)



# Show progress bar for frame processing
n = 1
for i, frame_ts in enumerate(tqdm(frame_timestamps, desc='Annotating video frames')):
    ret, frame = cap.read()
    if not ret:
        break
    # Find which fixation corresponds to this frame
    try:
        row = fix_df[(fix_df['start_ts_ns'] <= frame_ts) & (fix_df['end_ts_ns'] >= frame_ts)].iloc[0]
        x, y = int(row['x']), int(row['y'])
        cv2.circle(frame, (x, y), 15, (0, 0, 255), 3)
        n += 1
        out.write(frame)
    except:
        out.write(frame)
    if n > 1000:
        break

cap.release()
out.release()
print(f'Annotated video saved to {OUTPUT_PATH}')

def reencode_with_ffmpeg(input_path, output_path):
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if exists
        '-i', input_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-movflags', '+faststart',
        output_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f'Re-encoded video saved to {output_path}')
    except Exception as e:
        print(f'ffmpeg re-encoding failed: {e}')

fixed_output = OUTPUT_PATH.replace('.mp4', '_fixed.mp4')
reencode_with_ffmpeg(OUTPUT_PATH, fixed_output)

# Remove the original (non-fixed) video file
try:
    os.remove(OUTPUT_PATH)
    print(f'Removed original video: {OUTPUT_PATH}')
except Exception as e:
    print(f'Could not remove original video: {e}')
