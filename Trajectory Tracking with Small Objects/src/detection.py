# some global constants here
import numpy as np
import torch
import pandas as pd
from torchvision.utils import draw_bounding_boxes

TRAGET_CLASS_ID = 32
CONFIDENCE_THRESHOLD = 0.1
RESULT_PATH = './results'


'''
    detect_keypoints:
    input:
        input_video_path:str

    output:
        keypoints:pandas.core.frame.DataFrame

        fields of keypoints:
            frame_id    :int
            class_id    :int
            class_name  :str
            x_min       :float
            y_min       :float
            x_max       :float
            y_max       :float
'''

def detect_keypoints(frames: np.ndarray, model_path: str) -> pd.core.frame.DataFrame:
    # os.makedirs(RESULT_PATH, exist_ok=True)
    # model = torch.hub.load("ultralytics/yolov5", yolo_model, verbose=False)  # or yolov5n - yolov5x6, custom
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.iou = 0.7  # Intersection over Union threshold (default: 0.45)

    results = {}
    results['frame_id'] = []
    results['ball_detcted'] = []
    results['ball_coor_xmin'] = []
    results['ball_coor_ymin'] = []
    results['ball_coor_xmax'] = []
    results['ball_coor_ymax'] = []

    for i in range(frames.shape[0]):
        print(f"round: {i}")

        frame = frames[i]
        inference_results = model(frame)
        bbox_info = inference_results.pandas().xyxy[0]
        filtered_bboxes = bbox_info[bbox_info['confidence'] >= CONFIDENCE_THRESHOLD]

        results['frame_id'].append(i)

        # results['ball_detcted'] and other fields
        detected_balls_df = filtered_bboxes[filtered_bboxes["class"] == TRAGET_CLASS_ID]
        if detected_balls_df.empty:
            results['ball_detcted'].append(0)
            results['ball_coor_xmin'].append(-1.0)
            results['ball_coor_ymin'].append(-1.0)
            results['ball_coor_xmax'].append(-1.0)
            results['ball_coor_ymax'].append(-1.0)
        else:
            max_index = detected_balls_df["confidence"].idxmax()
            detected_balls_df = detected_balls_df.loc[max_index]

            results['ball_detcted'].append(1)
            results['ball_coor_xmin'].append(detected_balls_df["xmin"])
            results['ball_coor_ymin'].append(detected_balls_df["ymin"])
            results['ball_coor_xmax'].append(detected_balls_df["xmax"])
            results['ball_coor_ymax'].append(detected_balls_df["ymax"])

            # Draw bounding boxes
            drawing_box = torch.tensor([[detected_balls_df["xmin"], detected_balls_df["ymin"],
                                         detected_balls_df["xmax"], detected_balls_df["ymax"]]])
            image = frame.transpose(2, 0, 1)
            image = torch.tensor(image)

            image = draw_bounding_boxes(image, drawing_box, labels=None, colors=None, width=2)
            image = image.numpy()
            image = image.transpose(1, 2, 0)

            # cv2.imwrite(os.path.join(RESULT_PATH, f"output_frame_id_{i:05d}.jpg"), image)

    df_results = pd.DataFrame(results)
    return df_results

def pick_valid_keypoints(start_index, frame_batch_size, keypoints_df):
    # for index in range(start_index, min(start_index + frame_batch_size + 1, len(keypoints_df))):
    for index in range(start_index, min(start_index + frame_batch_size, len(keypoints_df))):
        key_point_df = keypoints_df.iloc[index]
        print(key_point_df['ball_detcted'])
        if key_point_df['ball_detcted'] == 1:
            return key_point_df
    return None
