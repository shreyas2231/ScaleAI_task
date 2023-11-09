from PIL import Image, ImageStat
import requests
import os
import scaleapi
import json
from sklearn.cluster import KMeans
from collections import Counter
import math
import numpy as np
from collections import defaultdict
import pytesseract
import numpy as np
from ultralytics import YOLO
from io import BytesIO

DISTANCE_THRESHOLD = 50


def calculate_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the intersection over union value
    return iou


# Function to find the best matches based on boxes distance

def find_closest_boxes_matches(detections, annotations, DISTANCE_THRESHOLD):
    matches = []
    for det in detections:
        det_box = (det[0], det[1], det[2], det[3])
        best_distance = float('inf')
        best_ann = None
        for ann in annotations:
            ann_box = (ann['left'], ann['top'], ann['left'] +
                       ann['width'], ann['top'] + ann['height'])
            distance = boxes_distance(det_box, ann_box)
            if distance < best_distance:
                best_distance = distance
                best_ann = ann
        if best_distance <= DISTANCE_THRESHOLD:
            matches.append((det, best_ann, best_distance))
    return matches


def boxes_distance(boxA, boxB):
    # Calculate the center of each box
    centerA = ((boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2)
    centerB = ((boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2)

    # Calculate Euclidean distance between the centers
    distance = ((centerA[0] - centerB[0]) ** 2 +
                (centerA[1] - centerB[1]) ** 2) ** 0.5
    return distance


def filter_traffic_light_annotations(annotations, traffic_light_label='traffic_control_sign'):
    """
    Filters annotations to include only those that are labeled as traffic lights.

    Args:
    - annotations (list of dicts): The list containing annotation dictionaries.
    - traffic_light_label (str): The label that is used in annotations to denote traffic lights.

    Returns:
    - list of dicts: A list containing only the annotations for traffic lights.
    """
    # Filter out annotations that are not labeled as traffic lights
    traffic_light_annotations = [
        ann for ann in annotations if ann['label'].lower() == traffic_light_label.lower()
    ]
    return traffic_light_annotations


# Function to find the best matches based on IoU
def find_best_iou_matches(detections, annotations, iou_threshold):
    matches = []
    for ann in annotations:
        ann_box = (ann['left'], ann['top'], ann['left'] +
                   ann['width'], ann['top'] + ann['height'])
        best_iou = 0
        best_det = None
        for det in detections:
            det_box = (det[0], det[1], det[2], det[3])
            iou = calculate_iou(det_box, ann_box)
            if iou > best_iou:
                best_iou = iou
                best_det = det
        if best_iou > iou_threshold:
            matches.append((best_det, ann, best_iou))
    return matches


# Helper function to download the image

def download_image(image_url, save_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download image from {image_url}")


# Main function to perform quality checks
def perform_quality_checks_advanced(tasks, model, iou_threshold):
    issues = []
    for task_obj in tasks:
        task = task_obj.as_dict()  # Convert the Task object to a dictionary
        task_id = task['task_id']  # Now you can access task_id
        # Extract image URL from the task params
        image_url = task['params']['attachment']
        # Define the path for the downloaded image
        image_path = f"../downloaded_images/{task_id}.jpg"

        # Download the image
        download_image(image_url, image_path)
        image = Image.open(image_path)
        annotations = filter_traffic_light_annotations(task['response']['annotations'] if 'response' in task else [
        ])  # Implement this function

        # Run model detection
        model_results = model.predict(image)
        model_detects = model_results[0]
        detections = model_detects.boxes.data.cpu().numpy()
        # just keep traffic signs
        filtered_detections = [d for d in detections if d[-1] == 9]

        print(filtered_detections)
        # Match detections with annotations
        # matches = find_best_iou_matches(
        # filtered_detections, annotations, iou_threshold)
        matches = find_closest_boxes_matches(
            detections, annotations, DISTANCE_THRESHOLD)

        # Generate issues for low IoU matches
        for det, ann, dist in matches:
            ann_box = (ann['left'], ann['top'], ann['left'] +
                       ann['width'], ann['top'] + ann['height'])
            det_box = (det[0], det[1], det[2], det[3])
            iou = calculate_iou(det_box, ann_box)
            if iou < 0.1:
                # If IoU is below 0.1, log it as an error.
                issues.append({
                    'task_id': task['task_id'],
                    'annotation_uuid': ann['uuid'],
                    'label': ann['label'],
                    'issue_type': 'error',
                    'message': f'Very low IoU ({iou:.2f}) indicates possible misannotation or detection failure.',
                    'iou': iou
                })
            elif iou < 0.9:
                # If IoU is between 0.1 and 0.9, log it as a warning.
                issues.append({
                    'task_id': task['task_id'],
                    'annotation_uuid': ann['uuid'],
                    'label': ann['label'],
                    'issue_type': 'warning',
                    'message': f'Moderate IoU ({iou:.2f}) between model detection and annotation for traffic_control_sign.',
                    'iou': iou
                })
    return issues
