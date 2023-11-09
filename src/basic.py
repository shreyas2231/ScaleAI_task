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


# Define the expected colors dictionary with RGB tuples instead of sets
COLOR_DICT = {
    "white": (255, 255, 255),
    "red": (204, 2, 2),
    "orange": (255, 150, 0),
    "yellow": (255, 235, 0),
    "green": (48, 132, 70),
    "blue": (67, 133, 255),
    "grey": (128, 128, 128),
    "black": (0, 0, 0)
}


def check_information_sign_text(image, bbox, label, task_id, annotation_uuid):
    """
    Check for text in an information sign and raise an error if none is found.

    Args:
    - image: PIL Image object
    - bbox: A tuple (left, top, right, bottom) representing the bounding box
    - label: The label of the object in the bounding box
    - task_id: The ID of the task
    - annotation_uuid: The UUID of the annotation

    Returns:
    - issues: A list of potential issues with the information sign
    """

    issues = []

    # Only apply this check to information signs
    if label.lower() == 'information_sign':
        # Crop the image to the bounding box
        cropped_image = image.crop(bbox)

        # Use pytesseract to do OCR on the cropped image
        text = pytesseract.image_to_string(cropped_image, lang='eng')

        # Check if any text was found
        if not text.strip():
            issues.append({
                'task_id': task_id,
                'annotation_uuid': annotation_uuid,
                'label': label,
                'issue_type': 'warning',
                'message': 'No text found in information sign.'
            })

    return issues


def check_traffic_sign_location(image, bbox, label, task_id, annotation_uuid):
    """
    Check if traffic signs are located in the top half of the image.

    Args:
    - image: PIL Image object
    - annotation_uuid: The UUID of the annotation
    - bbox: A tuple (left, top, right, bottom) representing the bounding box
    - label: The label of the object in the bounding box
    - task_id: The ID of the task

    Returns:
    - issues: A list of potential issues with the location of the traffic sign
    """
    issues = []

    # Only apply this check to traffic signs
    if label.lower() not in ['traffic_control_sign', 'information_sign', 'policy_sign']:
        return issues

    image_height = image.height
    bbox_center_y = (bbox[1] + bbox[3]) / 2
    # Check if the center of the bounding box is not in the top half of the image
    if bbox_center_y > image_height*0.8:
        issues.append({
            'task_id': task_id,
            'annotation_uuid': annotation_uuid,
            'label': label,
            'issue_type': 'warning',
            'message': 'Traffic sign is not located in the top half of the image.',
            'bbox_center_y': bbox_center_y
        })

    return issues


def check_background_color(image, bbox, label, task_id, annotation_uuid, annotation_bgcolor):
    """
    Check if the dominant color within the bounding box matches the expected background color.

    Args:
    - task_id: The ID of the task
    - annotation_uuid: The UUID of the annotation
    - image: PIL Image object
    - bbox: A tuple (left, top, right, bottom) representing the bounding box
    - annotation_bgcolor: The expected background color label for the annotation
    - label: The label of the object in the bounding box

    Returns:
    - issues: A list of potential issues with the background color mismatch
    """

    # Skip the check if the background color is not applicable
    if annotation_bgcolor.lower() in ['not_applicable', 'other']:
        return []

    issues = []
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')

    cropped_image = image.crop(bbox)
    np_image = np.array(cropped_image)

    # Reshape the image to be a list of pixels
    np_image = np.array(image.crop(bbox)).reshape((-1, 3))

    # Apply KMeans clustering to find dominant colors
    # We're looking for the most dominant color
    clt = KMeans(n_clusters=1, n_init=10)
    labels = clt.fit_predict(np_image)

    # Find the most common cluster center (dominant color)
    dominant_color = clt.cluster_centers_[0]
    # Convert to integer values
    dominant_color = [int(val) for val in dominant_color]

    # Calculate the Euclidean distance to each color in the COLOR_DICT
    min_rgb_distance = float('inf')
    dominant_color_label = ""
    for color, rgb_val in COLOR_DICT.items():
        distance = math.sqrt(
            sum([(dc - rc) ** 2 for dc, rc in zip(dominant_color, rgb_val)]))
        if distance < min_rgb_distance:
            min_rgb_distance = distance
            dominant_color_label = color

    # Adjust for grey being similar to white
    if dominant_color_label == 'grey':
        dominant_color_label = 'white'

    # Prepare a warning if there's a mismatch
    if dominant_color_label.lower() != annotation_bgcolor.lower():
        issues.append({
            'task_id': task_id,
            'annotation_uuid': annotation_uuid,
            'label': label,
            'issue_type': 'warning',
            'message': f'Expected {annotation_bgcolor} background but found {dominant_color_label}.',
            'dominant_color': dominant_color,
            'dominant_color_label': dominant_color_label
        })

    return issues


def check_low_intensity_and_variance_in_dark_image(image, bbox, label, task_id, annotation_uuid,
                                                   intensity_threshold=80,
                                                   variance_threshold=5):
    """
    Check if the bounding box is likely to be empty by analyzing the average pixel intensity
    and the variance of the pixel values within the bounding box,
    especially considering if the overall image is dark.

    Args:
    - image: PIL Image object
    - annotation_uuid: The UUID of the annotation
    - bbox: A tuple (left, top, right, bottom) representing the bounding box
    - label: The label of the object in the bounding box
    - task_id: The ID of the task
    - intensity_threshold: The threshold for average pixel intensity for the overall image to be considered dark
    - variance_threshold: The threshold for standard deviation within the bbox

    Returns:
    - issues: A list of potential issues with the bounding box being empty
    """
    issues = []

    # Convert image to grayscale for intensity analysis
    grayscale_image = image.convert("L")
    overall_stat = ImageStat.Stat(grayscale_image)
    average_intensity = overall_stat.mean[0]
    # Check the overall average pixel intensity to determine darkness
    if average_intensity < intensity_threshold:
        # Now let's check the intensity and variance inside the bounding box
        cropped_image = grayscale_image.crop(bbox)
        cropped_stat = ImageStat.Stat(cropped_image)
        cropped_average_intensity = cropped_stat.mean[0]
        cropped_std_dev = cropped_stat.stddev[0]
        # Check if the cropped area's intensity and variance are within acceptable parameters
        if cropped_average_intensity < intensity_threshold and cropped_std_dev < variance_threshold:
            issues.append({
                'task_id': task_id,
                'annotation_uuid': annotation_uuid,
                'label': label,
                'issue_type': 'warning',
                'message': 'Bounding box may be empty or indiscernible due to low intensity and variance in a dark image.',
                'bbox_average_intensity': cropped_average_intensity,
                'bbox_std_dev': cropped_std_dev
            })

    return issues


def check_unusual_bounding_box_size(image, bbox, label, task_id, annotation_uuid, min_size_threshold=0.00001, max_size_threshold=0.9):
    """
    Check if the bounding box is within acceptable size thresholds.

    Args:
    - image: PIL Image object
    - annotation_uuid: The UUID of the annotation
    - bbox: A tuple (left, top, right, bottom) representing the bounding box
    - label: The label of the object in the bounding box
    - task_id: The ID of the task
    - min_size_threshold: The minimum size ratio threshold of the bounding box relative to the image
    - max_size_threshold: The maximum size ratio threshold of the bounding box relative to the image

    Returns:
    - issues: A list of issues found with the bounding box sizes
    """
    issues = []
    image_width, image_height = image.size
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    # Calculate the size of the bounding box relative to the size of the image
    bbox_size_ratio = (bbox_width * bbox_height) / (image_width * image_height)

    # Check if the bounding box is too small
    if bbox_size_ratio < min_size_threshold:
        issues.append({
            'task_id': task_id,
            'label': label,
            'issue_type': 'warning',
            'message': f'Bounding box for {label} is too small.',
            'bbox_size_ratio': bbox_size_ratio
        })

    # Check if the bounding box is too large
    if bbox_size_ratio > max_size_threshold:
        issues.append({
            'task_id': task_id,
            'annotation_uuid': annotation_uuid,
            'label': label,
            'issue_type': 'Error',
            'message': f'Bounding box for {label} is too large.',
            'bbox_size_ratio': bbox_size_ratio
        })

    return issues


# Helper function to download the image


def download_image(image_url, save_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download image from {image_url}")

# Adjusted function to handle Task objects


def perform_quality_checks_basic(tasks):
    issues = []
    for task_obj in tasks:
        task = task_obj.as_dict()  # Convert the Task object to a dictionary
        task_id = task['task_id']  # Now you can access task_id
        # Extract image URL from the task params
        image_url = task['params']['attachment']
        annotations = task['response']['annotations'] if 'response' in task else [
        ]
        # Define the path for the downloaded image
        image_path = f"../downloaded_images/{task_id}.jpg"

        # Download the image
        download_image(image_url, image_path)

        if not os.path.exists(image_path):
            print(f"Failed to download image for task {task_id}")
            continue

        try:
            # Open the downloaded image
            with Image.open(image_path) as image:
                # Now perform checks based on annotations
                if 'response' in task and 'annotations' in task['response']:
                    for annotation in task['response']['annotations']:
                        bbox = (annotation['left'], annotation['top'], annotation['left'] +
                                annotation['width'], annotation['top'] + annotation['height'])
                        label = annotation['label']
                        annotation_uuid = annotation['uuid']
                        annotation_bgcolor = annotation['attributes'].get(
                            'background_color', '').lower()
                        issues.extend(check_unusual_bounding_box_size(
                            image, bbox, label, task_id, annotation_uuid))
                        issues.extend(check_low_intensity_and_variance_in_dark_image(
                            image, bbox, label, task_id, annotation_uuid))
                        issues.extend(check_background_color(
                            image, bbox, label, task_id, annotation_uuid, annotation_bgcolor))
                        issues.extend(check_traffic_sign_location(
                            image, bbox, label, task_id, annotation_uuid))
                        issues.extend(check_information_sign_text(
                            image, bbox, label, task_id, annotation_uuid))

                        pass

        except IOError:
            print(f"Could not open image for task {task_id}")

        # Clean up and remove the downloaded image if no longer needed
        os.remove(image_path)

    return issues
