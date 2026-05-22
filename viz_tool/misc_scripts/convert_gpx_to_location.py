#!/usr/bin/env python3
"""
Convert GPX file to Location.csv format for visualization tool.
"""

import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
import csv
import sys


def parse_gpx_to_location_csv(gpx_file, output_csv):
    """
    Parse GPX file and convert to Location.csv format.
    
    GPX format typically contains trackpoints with:
    - lat, lon, ele (elevation)
    - time
    
    Location.csv format requires:
    - time (nanoseconds timestamp)
    - seconds_elapsed
    - altitude
    - speedAccuracy
    - bearingAccuracy
    - latitude
    - altitudeAboveMeanSeaLevel
    - bearing
    - horizontalAccuracy
    - verticalAccuracy
    - longitude
    - speed
    """
    
    # Parse GPX file
    tree = ET.parse(gpx_file)
    root = tree.getroot()
    
    # Handle namespace
    namespace = {'default': 'http://www.topografix.com/GPX/1/1'}
    if root.tag.startswith('{'):
        ns_end = root.tag.index('}')
        namespace['default'] = root.tag[1:ns_end]
    
    # Extract trackpoints
    trackpoints = []
    
    # Try to find trackpoints in different GPX structures
    for trkpt in root.findall('.//default:trkpt', namespace):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        
        # Extract elevation
        ele_elem = trkpt.find('default:ele', namespace)
        altitude = float(ele_elem.text) if ele_elem is not None else 0.0
        
        # Extract time
        time_elem = trkpt.find('default:time', namespace)
        if time_elem is not None:
            time_str = time_elem.text
            # Parse ISO 8601 format
            try:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                timestamp_ns = int(dt.timestamp() * 1e9)
            except:
                timestamp_ns = 0
        else:
            timestamp_ns = 0
        
        # Extract extensions if available (for speed, accuracy, etc.)
        speed = 0.0
        horizontal_accuracy = 10.0  # Default values
        vertical_accuracy = 10.0
        bearing = 0.0
        speed_accuracy = 1.0
        bearing_accuracy = 180.0
        
        # Try to extract extension data
        extensions = trkpt.find('default:extensions', namespace)
        if extensions is not None:
            for ext_child in extensions:
                tag = ext_child.tag.split('}')[-1].lower()
                if 'speed' in tag:
                    try:
                        speed = float(ext_child.text)
                    except:
                        pass
                elif 'course' in tag or 'bearing' in tag:
                    try:
                        bearing = float(ext_child.text)
                    except:
                        pass
                elif 'hdop' in tag or 'horizontalaccuracy' in tag:
                    try:
                        horizontal_accuracy = float(ext_child.text)
                    except:
                        pass
                elif 'vdop' in tag or 'verticalaccuracy' in tag:
                    try:
                        vertical_accuracy = float(ext_child.text)
                    except:
                        pass
        
        trackpoints.append({
            'timestamp_ns': timestamp_ns,
            'latitude': lat,
            'longitude': lon,
            'altitude': altitude,
            'speed': speed,
            'bearing': bearing,
            'horizontal_accuracy': horizontal_accuracy,
            'vertical_accuracy': vertical_accuracy,
            'speed_accuracy': speed_accuracy,
            'bearing_accuracy': bearing_accuracy
        })
    
    if not trackpoints:
        print("Error: No trackpoints found in GPX file", file=sys.stderr)
        return False
    
    # Sort by timestamp
    trackpoints.sort(key=lambda x: x['timestamp_ns'])
    
    # Calculate seconds_elapsed
    start_time_ns = trackpoints[0]['timestamp_ns']
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'time', 'seconds_elapsed', 'altitude', 'speedAccuracy', 'bearingAccuracy',
            'latitude', 'altitudeAboveMeanSeaLevel', 'bearing', 'horizontalAccuracy',
            'verticalAccuracy', 'longitude', 'speed'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for pt in trackpoints:
            seconds_elapsed = (pt['timestamp_ns'] - start_time_ns) / 1e9
            
            row = {
                'time': pt['timestamp_ns'],
                'seconds_elapsed': seconds_elapsed,
                'altitude': pt['altitude'],
                'speedAccuracy': pt['speed_accuracy'],
                'bearingAccuracy': pt['bearing_accuracy'],
                'latitude': pt['latitude'],
                'altitudeAboveMeanSeaLevel': pt['altitude'],  # Same as altitude for GPX
                'bearing': pt['bearing'],
                'horizontalAccuracy': pt['horizontal_accuracy'],
                'verticalAccuracy': pt['vertical_accuracy'],
                'longitude': pt['longitude'],
                'speed': pt['speed']
            }
            writer.writerow(row)
    
    print(f"Converted {len(trackpoints)} trackpoints from {gpx_file} to {output_csv}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert GPX file to Location.csv format for visualization'
    )
    parser.add_argument('gpx_file', help='Input GPX file path')
    parser.add_argument('-o', '--output', help='Output CSV file path', 
                        default='Location_from_gpx.csv')
    
    args = parser.parse_args()
    
    try:
        success = parse_gpx_to_location_csv(args.gpx_file, args.output)
        if success:
            print(f"Successfully created: {args.output}")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
