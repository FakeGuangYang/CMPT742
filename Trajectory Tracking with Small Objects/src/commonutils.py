import torch
import math
from cotracker.utils.visualizer import Visualizer, read_video_from_path

"""
    get_performance_measurements_stats:
    
    @brief:
        calculate precision and recall

        1. Consider the visibility prediction:

        Predicted as Positive:              ball trajectory detected as visible.
        Predicted as Negative:              ball trajectory detected as invisible.
        Ground Truth as Positive:           ball is visible.
        Ground Truth as Negative:           ball is invisible.

        TP: ball is visible, ball trajectory detected as visible
        FP: ball is invisible, ball trajectory detected as visible.
        FN: ball is visible, ball trajectory detected as invisible.

        2. Consider the trajectory position prediction of [visible part]:

        Accurate Predict:                       predicted point is in the box.
        Mispredict or Inaccurate Predict:       predicted point is not in the box.

        divation:                               distance between predicted point and box center.

    @arguments:
        pred_tracks:
        pred_visibility:
        gt_label_path:

    @return:
        precision:  float
        recall:     float
        diviation:
"""

def get_performance_measurements_stats(all_pred_tracks, all_pred_visibility, width, height, all_gt_x_center, all_gt_y_center, all_gt_x_width, all_gt_y_height, all_gt_visibility):
    
    all_pred_tracks     = all_pred_tracks.squeeze(0).squeeze(1)
    all_pred_visibility = all_pred_visibility.squeeze(0).squeeze(1)
    all_pred_tracks[..., 0] = all_pred_tracks[..., 0]/width
    all_pred_tracks[..., 1] = all_pred_tracks[..., 1]/height
    
    CNT_VIS_TP = 0
    CNT_VIS_FP = 0
    CNT_VIS_FN = 0

    CNT_POS_T       = 0
    CNT_POS_F       = 0

    DIST_DIV        = []

    for pred_track, pred_visibility, gt_x_center, gt_y_center, gt_x_width, gt_y_height, gt_visibility in \
        zip(all_pred_tracks, all_pred_visibility, all_gt_x_center, all_gt_y_center, all_gt_x_width, all_gt_y_height, all_gt_visibility):

        # For visibility prediction
        if (pred_visibility == True) and (gt_visibility == True):
            CNT_VIS_TP += 1
        elif (pred_visibility == True) and (gt_visibility == False):
            CNT_VIS_FP += 1
        elif (pred_visibility == False) and (gt_visibility == True):
            CNT_VIS_FN += 1
        
        # For position prediction
        if (pred_visibility == True) and (gt_visibility == True):
            x_dev = math.fabs(gt_x_center - pred_track[0])
            y_dev = math.fabs(gt_y_center - pred_track[1])

            if(x_dev < gt_x_width) and (y_dev < gt_y_height):
                CNT_POS_T += 1
            else:
                CNT_POS_F += 1
            
            # divation        = x_dev ** 2 + y_dev ** 2
            divation        = math.sqrt(x_dev ** 2 + y_dev ** 2)
            DIST_DIV.append(divation)
    
    
    return CNT_VIS_TP, CNT_VIS_FP, CNT_VIS_FN, CNT_POS_T, CNT_POS_F, DIST_DIV



"""
    play_video:
        Loads a video path and play the video.
    Args:
        video_path: Path to the video file.
    Returns:
        None.
"""


def play_video(video_path):
    # open video
    cap = cv2.VideoCapture(video_path)

    # see if it is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # read frame by frame
        ret, frame = cap.read()
        if not ret:  # return if no more frame
            break

        # current frame
        cv2.imshow("Video", frame)

        # press q to quit playing
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release source
    cap.release()
    cv2.destroyAllWindows()


# show video on Colab
def show_video(video_path):
    # read video and encode to base64
    video_file = open(video_path, "rb").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width="640" height="480" autoplay loop controls><source src="{video_url}"></video>""")


"""
    load_video_as_images:
        Loads a video and extracts frames as a list of images.

    Args:
        video_path: Path to the video file.

    Returns:
        A list of images extracted from the video.
"""


def load_video_as_images(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def dump_video_to_frames(video_path, output_folder):
    """
    Extracts frames from a video and saves them as images in a specified folder.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where frames will be saved.

    Returns:
        int: Number of frames extracted and saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    frame_count = 0
    while True:
        # Read one frame
        success, frame = video_capture.read()
        if not success:
            break

        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {frame_count} frames to {output_folder}")
    return frame_count

def visualize_tracking(video, pred_tracks, pred_visibility, path_saving='./videos', name_saving='results', mode='cool'):
    # visualization
    DEFAULT_DEVICE = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Default device: ", DEFAULT_DEVICE)
    # allow fallback to cpu if occurs error with PyTorch
    if DEFAULT_DEVICE == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    pred_tracks = torch.from_numpy(pred_tracks)
    pred_visibility = torch.from_numpy(pred_visibility)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None]

    vis = Visualizer(
        save_dir=path_saving,
        fps=29,
        linewidth=4,
        mode=mode,
        tracks_leave_trace=-1
    )

    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename=name_saving)

    return

def test_dataset(datasest_path, input_detection_model_path, frame_batch_size=20, frame_batch_offset=10):
    cases = os.listdir(datasest_path)
    
    ALL_CNT_VIS_TP = 0
    ALL_CNT_VIS_FP = 0
    ALL_CNT_VIS_FN = 0
    ALL_CNT_POS_T = 0
    ALL_CNT_POS_F = 0
    ALL_DIST_DIV = []

    for case in cases:
        # load data
        test_data_path = os.path.join(datasest_path, case)
        video_frames, all_gt_x_center, all_gt_y_center, all_gt_x_width, all_gt_y_height, all_gt_visibility = load_data(test_data_path, f'{case}.mp4', 'labels')
        frame_cnt, height, width, channel_cnt = video_frames.shape
        
        # inference
        keypoints_df = detect_keypoints(video_frames, input_detection_model_path)
        all_pred_tracks, all_pred_visibility = tracking_all(keypoints_df, video_frames, frame_batch_size, frame_batch_offset)

        # visualize results
        visualize_tracking(video_frames, all_pred_tracks, all_pred_visibility, path_saving='./videos', name_saving=case)
    
        # get performance stats
        CNT_VIS_TP, CNT_VIS_FP, CNT_VIS_FN, CNT_POS_T, CNT_POS_F, DIST_DIV = get_performance_measurements_stats(all_pred_tracks, all_pred_visibility, width, height, all_gt_x_center, all_gt_y_center, all_gt_x_width, all_gt_y_height, all_gt_visibility)

        # accumulate performance stats
        ALL_CNT_VIS_TP  += CNT_VIS_TP
        ALL_CNT_VIS_FP  += CNT_VIS_FP
        ALL_CNT_VIS_FN  += CNT_VIS_FN
        ALL_CNT_POS_T   += CNT_POS_T
        ALL_CNT_POS_F   += CNT_POS_F
        ALL_DIST_DIV    += DIST_DIV
    
    PRECISION_VIS, RECALL_VIS, ACCURACY_POS, AVG_DIVATION = calculate_performance_matrics(ALL_CNT_VIS_TP, ALL_CNT_VIS_FP, ALL_CNT_VIS_FN, ALL_CNT_POS_T, ALL_CNT_POS_F, sum(ALL_DIST_DIV))
    return PRECISION_VIS, RECALL_VIS, ACCURACY_POS, AVG_DIVATION, ALL_DIST_DIV

