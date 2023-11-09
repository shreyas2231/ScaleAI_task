import json
import os
import shutil


def write_issues_to_json_file(issues, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(issues, json_file, indent=4, ensure_ascii=False)

    print(f"All issues have been written to {file_path}")


def restructure_issues(issues):
    structured_data = {}

    for issue in issues:
        task_id = issue.get('task_id')
        annotation_uuid = issue.get('annotation_uuid')

        # Skip issues that don't have both a task_id and an annotation_uuid
        if not task_id or not annotation_uuid:
            continue  # Optionally, log this case for debugging purposes

        # Initialize a dictionary for the task_id if it does not exist
        if task_id not in structured_data:
            structured_data[task_id] = {
                'annotations': {},
                'total_warnings': 0,
                'total_errors': 0,
            }

        # Initialize the entries for the annotation_uuid if it does not exist
        if annotation_uuid not in structured_data[task_id]['annotations']:
            structured_data[task_id]['annotations'][annotation_uuid] = {
                'issues': [],
                'warnings': 0,
                'errors': 0
            }

        # Append the issue to the list for the annotation_uuid
        structured_data[task_id]['annotations'][annotation_uuid]['issues'].append(
            issue)

        # Increment the warning or error counters
        if issue['issue_type'].lower() == 'warning':
            structured_data[task_id]['annotations'][annotation_uuid]['warnings'] += 1
            structured_data[task_id]['total_warnings'] += 1
        elif issue['issue_type'].lower() == 'error':
            structured_data[task_id]['annotations'][annotation_uuid]['errors'] += 1
            structured_data[task_id]['total_errors'] += 1

    # Assemble the final output with stats at the top
    final_output = {}
    for task_id, task_data in structured_data.items():
        num_annotations = len(task_data['annotations'])

        # Calculate averages
        average_warnings = task_data['total_warnings'] / \
            num_annotations if num_annotations else 0
        average_errors = task_data['total_errors'] / \
            num_annotations if num_annotations else 0

        # Start with stats
        final_structured_data = {
            'total_warnings': task_data['total_warnings'],
            'total_errors': task_data['total_errors'],
            'average_warnings': average_warnings,
            'average_errors': average_errors,
        }

        # Add annotations
        final_structured_data['annotations'] = [
            {'annotation_uuid': ann_uuid, **ann_data}
            for ann_uuid, ann_data in task_data['annotations'].items()
        ]

        # Assign to final output
        final_output[task_id] = final_structured_data

    return final_output


def write_summary_to_json_file(structured_data, file_path):
    # Initialize a dictionary to hold the summary data
    summary_data = {}

    # Loop through each task and calculate the summary
    for task_id, task_info in structured_data.items():
        total_warnings = task_info['total_warnings']
        total_errors = task_info['total_errors']
        total_annotations = len(task_info['annotations'])
        avg_warnings_per_ann = total_warnings / \
            total_annotations if total_annotations else 0
        avg_errors_per_ann = total_errors / total_annotations if total_annotations else 0

        # Store the calculations in the summary dictionary
        summary_data[task_id] = {
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'total_suspicious_annotations': total_annotations,
            'avg_warnings_per_ann': avg_warnings_per_ann,
            'avg_errors_per_ann': avg_errors_per_ann
        }

    # Write the summary data to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(summary_data, json_file, indent=4, ensure_ascii=False)

    print(f"Summary data has been written to {file_path}")


def clear_downloaded_images(directory="../downloaded_images"):
    # Check if the directory exists
    if os.path.exists(directory):
        # Remove all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
