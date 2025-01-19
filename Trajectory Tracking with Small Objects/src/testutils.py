import os
import matplotlib.pyplot as plt
import numpy as np

from dataloader import load_data_demo_only
from dataloader import load_data
from detection import detect_keypoints
from tracking import tracking_all
from commonutils import visualize_tracking
from commonutils import get_performance_measurements_stats

def calculate_performance_matrics(CNT_VIS_TP, CNT_VIS_FP, CNT_VIS_FN, CNT_POS_T, CNT_POS_F, TOTAL_DIVATION):
    PRECISION_VIS   = 0 if CNT_VIS_TP == 0 else float(CNT_VIS_TP) / float(CNT_VIS_TP + CNT_VIS_FN)
    RECALL_VIS      = 0 if CNT_VIS_TP == 0 else float(CNT_VIS_TP) / float(CNT_VIS_TP + CNT_VIS_FP)
    ACCURACY_POS    = 0 if CNT_POS_T == 0 else float(CNT_POS_T) / float(CNT_POS_T + CNT_POS_F)
    AVG_DIVATION    = 0 if CNT_VIS_TP == 0 else TOTAL_DIVATION / CNT_VIS_TP
    
    return PRECISION_VIS, RECALL_VIS, ACCURACY_POS, AVG_DIVATION

def stat_onevar_histogram(onevar_data, saving_path='histogram.jpg', title='Histogram of Data', x_label='data_value', y_label='frequency', bucket_width = 0.05):
    plt.figure()
    bin_edges = np.arange(0.0, 1.0 + bucket_width, bucket_width)
    plt.hist(onevar_data, bins=bin_edges, edgecolor='black')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(saving_path)

def test_demo(input_video_path, input_detection_model_path, frame_batch_size, frame_batch_offset, result_video_root, result_video_name):
    video_frames = load_data_demo_only(input_video_path)
    keypoints_df = detect_keypoints(video_frames, input_detection_model_path)
    all_pred_tracks, all_pred_visibility = tracking_all(keypoints_df, video_frames, frame_batch_size, frame_batch_offset)
    visualize_tracking(video_frames, all_pred_tracks, all_pred_visibility, path_saving=result_video_root, name_saving=result_video_name)

def test_dataset(datasest_path, input_detection_model_path, frame_batch_size=20, frame_batch_offset=10):
    cases = os.listdir(datasest_path)
    
    ALL_CNT_VIS_TP  = 0
    ALL_CNT_VIS_FP  = 0
    ALL_CNT_VIS_FN  = 0
    ALL_CNT_POS_T   = 0
    ALL_CNT_POS_F   = 0
    ALL_DIST_DIV    = []

    for case in cases:
        # load data
        test_data_path = os.path.join(datasest_path, case)
        video_frames, all_gt_x_center, all_gt_y_center, all_gt_x_width, all_gt_y_height, all_gt_visibility = load_data(test_data_path, f'{case}.mp4', 'labels')
        frame_cnt, height, width, channel_cnt = video_frames.shape
        
        # inference
        keypoints_df = detect_keypoints(video_frames, input_detection_model_path)
        all_pred_tracks, all_pred_visibility = tracking_all(keypoints_df, video_frames, frame_batch_size, frame_batch_offset)

        # visualize results
        visualize_tracking(video_frames, all_pred_tracks, all_pred_visibility, path_saving='./videos', name_saving=f"{case}_result_{frame_batch_size}_{frame_batch_offset}")
    
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

