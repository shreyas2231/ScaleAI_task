import json
from scaleapi import ScaleClient
from ultralytics import YOLO

from basic import (COLOR_DICT, check_information_sign_text,
                   check_traffic_sign_location, check_background_color, check_low_intensity_and_variance_in_dark_image,
                   check_unusual_bounding_box_size, perform_quality_checks_basic)


from advanced import (calculate_iou, boxes_distance, find_closest_boxes_matches,
                      filter_traffic_light_annotations, perform_quality_checks_advanced)

from json_helper import (write_issues_to_json_file,
                         restructure_issues, write_summary_to_json_file, clear_downloaded_images)

# Configuration
PROJECT_NAME = "Traffic Sign Detection"
model = YOLO("../model/yolov8n.pt")

IOU_THRESHOLD = 0.1
DISTANCE_THRESHOLD = 50
CHECK_MODE = "both"  # Toggle between 'basic', 'advanced' and 'both'.

# Initialize the client
try:
    client = ScaleClient(API_KEY)
    # Get tasks
    try:
        tasks = client.tasks(project=PROJECT_NAME)
    except Exception as e:
        print(f"Failed to retrieve tasks: {e}")
except Exception as e:
    print(f"Failed to initialize the Scale API client: {e}")

all_issues = []

# Perform checks based on the mode
if CHECK_MODE == "basic":
    all_issues = perform_quality_checks_basic(tasks)
elif CHECK_MODE == "advanced":
    all_issues = perform_quality_checks_advanced(tasks, model, IOU_THRESHOLD)
elif CHECK_MODE == "both":
    all_issues = perform_quality_checks_basic(tasks)
    all_issues.extend(
        perform_quality_checks_advanced(tasks, model, IOU_THRESHOLD))
else:
    raise ValueError(f"Unknown check mode: {CHECK_MODE}")

# Rearrange the issues into the desired structure
restructured_data = restructure_issues(all_issues)

# Write the issues to a JSON file
write_issues_to_json_file(restructured_data, '../results/quality_issues.json')

# Write a summary to a JSON file
write_summary_to_json_file(restructured_data, '../results/summary_data.json')

# Remove all the downloaded images
clear_downloaded_images()
