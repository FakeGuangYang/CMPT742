import os
from cotracker.utils.visualizer import Visualizer, read_video_from_path

"""
    Reads TXT files, extracts the last four values from each line,
    or assigns zero for empty files, and stores them into separate lists.

    Parameters:
    - gt_label_path (str): Path to the labedingTXT files.

    Returns:
    - Four lists containing the extracted values from all files.
    - visibility

"""
def read_from_ground_truth_label(gt_label_path):
    
    # Sort files based on the 4 digits after "frame_-"
    label_files = [f for f in os.listdir(gt_label_path) if f.endswith(".txt")]
    label_files.sort(key=lambda x: int(x.split("frame_")[1][:4]))

    # Initialize lists to store the extracted values
    x_center, y_center, x_width, y_height = [], [], [], []
    visibility = []

    # Process each file
    for label_file in label_files:
        label_file_path = os.path.join(gt_label_path, label_file)
        with open(label_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # assumption: there is at most 1 ball groundtruth in the pic.
            # print(lines)
            # assert(len(lines) <= 1)

            # With ball object 
            if lines:
                ground_truth_line = lines[0].split()
                assert(len(ground_truth_line) == 5)  # Ensure there are exactly 5 values
                
                x_center.append(float(ground_truth_line[1]))
                y_center.append(float(ground_truth_line[2]))
                x_width.append(float(ground_truth_line[3]))
                y_height.append(float(ground_truth_line[4]))

                visibility.append(True)

            # Without ball object 
            else:
                x_center.append(0)
                y_center.append(0)
                x_width.append(0)
                y_height.append(0)

                visibility.append(False)

    return x_center, y_center, x_width, y_height, visibility

def load_data_demo_only(input_video_path):
    video_frames = read_video_from_path(input_video_path)
    return video_frames

def load_data(test_data_path, video_name, label_folder_name):
    video_frames = read_video_from_path(os.path.join(test_data_path, video_name))
    all_gt_x_center, all_gt_y_center, all_gt_x_width, all_gt_y_height, all_gt_visibility = \
        read_from_ground_truth_label(os.path.join(test_data_path, label_folder_name))

    return video_frames, all_gt_x_center, all_gt_y_center, all_gt_x_width, all_gt_y_height, all_gt_visibility
