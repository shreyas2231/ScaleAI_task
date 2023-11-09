# Annotation Quality Checks

This repository contains tools for performing quality checks on annotated images, primarily focusing on traffic sign detection tasks. The checks range from basic to advanced, aiming to identify and flag potential annotation errors.

Find my reflections on analysing possible advancements with more time and probable future of these annotation checks in the [Reflections](./Reflections.md) file.

## Overview

The quality check process is designed to ensure the integrity and accuracy of image annotations, particularly in the context of traffic sign detection. Leveraging the Scale API, this automated system utilizes an API key to fetch tasks and corresponding images for analysis. Once the images are downloaded, they undergo a series of rigorous checks aimed at identifying any discrepancies or errors in the annotations.

These checks are categorized into two main types:

1. **Basic Checks**: These are simpler, rule-based validations that catch common annotation mistakes such as incorrectly sized bounding boxes, annotations in unlikely image locations, or missing text in information signs.

2. **Advanced Checks**: These involve more sophisticated methods, including the use of YOLOv8 object detection to compare automated detections with existing annotations, enhancing the potential to spot nuanced errors.

Upon completion of the checks, annotations that fail to meet the quality standards are recorded. The results are then compiled into two types of JSON files:

- **Detailed Review File**: Contains a comprehensive list of issues for each annotation, providing an in-depth analysis for reviewers to address specific errors.

- **Summary File**: Offers a high-level overview of the quality check results, including total warnings, total errors, and average issues per annotation, enabling a quick assessment of the annotation batch's overall quality.

Through this automated and methodical approach, the quality check system streamlines the review process, ensuring that only the highest quality annotations are utilized.

## Requirements

To run the code in this repository, you will need:

- Python 3.10
- pip (Python package installer)

Ensure you have the latest version of pip installed:

```bash
pip install --upgrade pip

```

Install the required Python libraries using pip:

```bash
pip install numpy pillow pytesseract sklearn scaleapi ultralytics

```

## Project File Structure

Below is the structure of the repository:

```sh

Scale Assignment/
├── README.md
├── Reflections.md (Further improvements and analysis on the future of annotation checks)
├── downloaded_images/
│   └── ... (downloaded images are stored here)
├── model/
│   └── yolov8n.pt (YOLOv8n model file)
├── results/
│   ├── quality_issues.json (detailed quality issues)
│   └── summary_data.json (summary of issues)
└── src/
    ├── __pycache__/
    ├── advanced.py (advanced quality check scripts)
    ├── basic.py (basic quality check scripts)
    ├── json_helper.py (helper scripts for JSON operations)
    └── main.py (main script to run quality checks)
```

## How to Run the Code

Firstly, clone the repo:

```bash
git clone https://github.com/shreyas2231/ScaleAI_task.git

```

then headover to `src/` by running:

```bash
cd src/

```

To execute the quality checks, run the `main.py` file. You can toggle between 'basic', 'advanced', or 'both' check modes by setting the `CHECK_MODE` variable at the top of the `main.py` script.

```bash
python main.py

```

## Thought Process

The development of quality checks began by identifying common causes of poor annotations:

- Incorrectly labeled annotations.
- Annotations made in haste leading to incorrect bounding boxes.
- Mislabeling or confusing one sign with one another.

### Basic Checks

1. **Unusual Bounding Box Sizes**: Function `check_unusual_bounding_box_size` flags bounding boxes that are too small or large compared to the image size, indicating potential annotation errors.

2. **Dark Area Annotation**: The `check_low_intensity_and_variance_in_dark_image` function analyzes if a dark area in an image is likely to be a legitimate object or just an unannotable dark region, especially in already dark images.

3. **Traffic Lights at the Bottom of the Image**: `check_traffic_sign_location` looks for traffic light annotations in unlikely image locations, such as the bottom half of the image, which could point to inaccuracies unless the vehicle's trajectory justifies it.

4. **Dominant Background Color Check**: By finding the dominant color within a bounding box (`check_background_color`) and comparing it to the expected background color, mismatches can be identified, suggesting incorrect annotations.

5. **Text Detection in Information Signs**: Utilizing `pytesseract` OCR, the `check_information_sign_text` function checks for the presence of text within information signs, raising an error if none is found.

### Advanced Checks

To address inaccurately bounded boxes, `perform_quality_checks_advanced` uses YOLOv8 to detect traffic lights and compares the overlap with existing annotations. This could be enhanced by training a custom model to detect specific traffic signs.
