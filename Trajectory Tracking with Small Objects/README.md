# CMPT742
CMPT742 final proj

# Start:

```bash
git clone https://github.com/FakeGuangYang/CMPT742.git &&
cd cd Trajectory\ Tracking\ with\ Small\ Objects &&
git submodule update --init --recursive
```

# Set Up Environment
1. Install ``anaconda``
2. create conda env using
    ```bash
    conda env create -f environment.yml
    ```
3. Install ``pip``
4. install pip packages using
    ```bash
    pip install -r requirements.txt
    ```
5. Install ``co-tracker`` locally
    ```bash
    cd co-tracker
    pip install -e .
    cd ..
    ```
# Data
use video in `./videos` to demo
unpacked `annotated_data.zip`, use `annotated_data` as test dataset

# Run
## Run only one video as demo
```bash
python3 src/main.py --mode=demo --input_video_path=./videos/test_video1.mp4 --input_detection_model_path=./models/yolov5s.pt --batch_size=60 --overlap_size=10 --result_video_root=./videos --result_video_name=demo_results
```

## Run with a testset
```bash
python3 src/main.py --mode=test --dataset_path=./annotated_data --input_detection_model_path=./models/yolov5s.pt
```

## Reference:
[yolo-v5]       git@github.com:ultralytics/yolov5.git \
[co-tracker]    git@github.com:facebookresearch/co-tracker.git
