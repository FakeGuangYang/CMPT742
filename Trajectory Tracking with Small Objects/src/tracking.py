import torch
import os
import numpy as np

from cotracker.predictor import CoTrackerPredictor
from detection import pick_valid_keypoints

co_tracker_model_path = os.path.join(os.getcwd(), 'co-tracker/checkpoints/scaled_offline.pth')

def translate_keypoint_to_query(start_frame_index, keypoint_df):
    query = None

    if keypoint_df is not None:
        query = []

        frame_id = keypoint_df['frame_id'] - start_frame_index
        x_min = keypoint_df['ball_coor_xmin']
        x_max = keypoint_df['ball_coor_xmax']
        y_min = keypoint_df['ball_coor_ymin']
        y_max = keypoint_df['ball_coor_ymax']

        query.append([frame_id, (x_min + x_max) / 2, (y_min + y_max) / 2])

    return query


def translate_track_vis_to_query(start_index_offset, pred_tracks, pred_visibility):
    if (pred_tracks is not None) and (pred_visibility is not None):
        # print(pred_tracks)
        # print(pred_tracks.shape)
        # print(pred_visibility)
        # print(pred_visibility.shape)

        vis = pred_visibility.squeeze(-1).squeeze(0)
        last_tracking_frame_id_last = torch.where(vis == True)[-1][-1].item()

        pred_tracks = pred_tracks.squeeze(0)
        x_coor = pred_tracks[last_tracking_frame_id_last][0][0].item()
        y_coor = pred_tracks[last_tracking_frame_id_last][0][1].item()
        tracking_frame_id = last_tracking_frame_id_last - start_index_offset

        if (tracking_frame_id < 0):
            print("nothing usable in last tracking")
            return None

        query = []
        print(f"DEBUG: from translate_track_vis_to_query: query = {query}")
        query.append([tracking_frame_id, x_coor, y_coor])
        return query

    return None


def concat_track_vis(all_pred_tracks, all_pred_visibility, total_frame_cnt, frame_batch_size, frame_batch_offset):
    final_pred_tracks = np.zeros((1, total_frame_cnt, 1, 2), dtype=np.float32)
    final_pred_visibility = np.zeros((1, total_frame_cnt, 1), dtype=bool)

    start_index = 0

    for pred_track, pred_visibility in zip(all_pred_tracks, all_pred_visibility):
        if (pred_track is not None):
            pred_track = pred_track.cpu().numpy()
            pred_visibility = pred_visibility.cpu().numpy()

            max_frame = min(start_index + frame_batch_size, start_index + pred_track.shape[1])

            final_pred_tracks[:, start_index: max_frame, :, :] = pred_track
            final_pred_visibility[:, start_index: max_frame, :] = pred_visibility

        start_index += frame_batch_size
        start_index -= frame_batch_offset

    return final_pred_tracks, final_pred_visibility


'''

'''


def tracking_all(keypoints_df, video, frame_batch_size=20, frame_batch_offset=10):
    frame_cnt_total = video.shape[0]
    frame_index_start = 0
    last_frame_index_start = 0
    last_pred_track = None
    last_pred_visibility = None

    all_pred_tracks = []
    all_pred_visibility = []

    while (frame_index_start < frame_cnt_total - 1):

        print(f"Debug: frame_index_start = {frame_index_start}")

        # pre-screen query points
        query_from_detection_df     = pick_valid_keypoints(frame_index_start, frame_batch_size, keypoints_df)
        query_from_detection        = translate_keypoint_to_query(frame_index_start, query_from_detection_df)
        query_from_last_tracking    = translate_track_vis_to_query(frame_batch_size - frame_batch_offset, last_pred_track, last_pred_visibility)
        query                       = query_from_last_tracking if (query_from_last_tracking != None) else query_from_detection
        
        print(f"Debug3: query_from_detection_df = {query_from_detection_df}")
        print(f"Debug3: query_from_detection = {query_from_detection}")
        print(f"Debug3: query_from_last_tracking = {query_from_last_tracking}")
        print(f"Debug3: query = {query}")

        # inference using tracking_batch
        pred_tracks = None
        pred_visibility = None
        if (query != None):
            print(f"Debug: frame_index_start = {frame_index_start}")
            print(f"Debug: frame_index_start + frame_batch_size = {frame_index_start + frame_batch_size}")

            video_batch = video[frame_index_start: frame_index_start + frame_batch_size, :, :, :]
            pred_tracks, pred_visibility = tracking_batch(video_batch, query)

            print(f"DEBUG2: round pred_tracks = {pred_tracks}")
            print(f"DEBUG2: round pred_visibility = {pred_visibility}")

        # store results
        all_pred_tracks.append(pred_tracks)
        all_pred_visibility.append(pred_visibility)

        # step
        last_frame_index_start = frame_index_start
        last_pred_track = pred_tracks
        last_pred_visibility = pred_visibility
        frame_index_start += frame_batch_size
        frame_index_start -= frame_batch_offset

    all_pred_tracks, all_pred_visibility = concat_track_vis(all_pred_tracks, all_pred_visibility, frame_cnt_total,
                                                            frame_batch_size, frame_batch_offset)

    return all_pred_tracks, all_pred_visibility


'''
    tracking_batch:
        Load the input video and track its trajectories.
    input:
        input_video_path:str
        query           :list[frame_number, x_center, y_center]
    output:
        output_video_path:str
'''


def tracking_batch(video, query):
    DEFAULT_DEVICE = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Default device: ", DEFAULT_DEVICE)
    # allow fallback to cpu if occurs error with PyTorch
    if DEFAULT_DEVICE == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # initialize model
    model = CoTrackerPredictor(checkpoint=co_tracker_model_path).to(DEFAULT_DEVICE)

    # add query points
    queries = torch.tensor(query).to(DEFAULT_DEVICE, dtype=torch.float32)

    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].to(DEFAULT_DEVICE, dtype=torch.float32)

    print(f"Debug: tracking_batch: query = {queries[None]}")
    print(f"Debug: tracking_batch: video.shape = {video.shape}")
    print(f"Debug: video dtype = {video.dtype}")

    # np.savetxt("queries_1.txt", queries.cpu().numpy())
    # np.savetxt("video_1.txt", video.reshape(-1, video.shape[-1]).cpu().numpy())

    print(f"Debug: BEFORE INFERENCE: queries={queries}")

    pred_tracks, pred_visibility = model(video, queries=queries[None])
    return pred_tracks, pred_visibility
