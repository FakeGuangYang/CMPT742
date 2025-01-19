import argparse
import os

from testutils import test_demo
from testutils import test_dataset
from testutils import stat_onevar_histogram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--input_detection_model_path', type=str, required=True)
    
    parser.add_argument('--input_video_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--overlap_size', type=int)
    parser.add_argument('--result_video_root', type=str)
    parser.add_argument('--result_video_name', type=str)

    args = parser.parse_args()
    
    if args.mode == "demo":
        test_demo(args.input_video_path, args.input_detection_model_path, args.batch_size, args.overlap_size, args.result_video_root, args.result_video_name)
    elif args.mode == "test":
        for frame_batch_size in range(20, 101, 10):
            PRECISION_VIS, RECALL_VIS, ACCURACY_POS, AVG_DIVATION, ALL_DIST_DIV = test_dataset(args.dataset_path, args.input_detection_model_path, frame_batch_size=frame_batch_size, frame_batch_offset=10)

            with open("result_matrics.txt", "a") as writer:
                writer.write(f"RESULTS: with yolo, frame_batch_size = {frame_batch_size}, overlap_frame_size = 10\n")
                writer.write(f"OVERALL PRECISION_VIS   =  {PRECISION_VIS}\n")
                writer.write(f"OVERALL RECALL_VIS   =  {RECALL_VIS}\n")
                writer.write(f"OVERALL ACCURACY_POS   =  {ACCURACY_POS}\n")
                writer.write(f"OVERALL AVG_DIVATION   =  {AVG_DIVATION}\n")

            if(frame_batch_size == 100):
                stat_onevar_histogram(ALL_DIST_DIV, saving_path=f'deviation_dist_overall_{frame_batch_size}_10.png', title=f'deviation_distribution co-tracker-only', x_label='deviation - ratio to width and height', y_label='frequency - times')
            else:
                stat_onevar_histogram(ALL_DIST_DIV, saving_path=f'deviation_dist_overall_{frame_batch_size}_10.png', title=f'deviation_distribution frame_batch_size = {frame_batch_size}', x_label='deviation - ratio to width and height', y_label='frequency - times')

if __name__ == '__main__':
    main()
