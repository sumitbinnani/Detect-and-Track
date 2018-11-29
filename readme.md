# End to End  Detection and Tracking

The repository contains code for detection and tracking. The code uses Deep Learning Detectors and  Kalman Filter for tracking.

## Usage
```
python main.py --detector yolov3 --input <input_video> --output <output_path>
```

## Setup

```
git clone 

# to use yolo-v3 detector
git submodule update --init --recursive

# download yolo-v3 weights
cd detection/yolov3
mkdir weights
cd weights
wget https://pjreddie.com/media/files/yolov3.weights 
```



## Detectors

Currently, module supports two detectors:
- Mobilenet Single Shot Detector
- Yolo-v3

One can implement their own detector by extending `BaseDetector` class defined [here](detection/base_detector.py)

## Trackers

Currently, module supports only Kalman Filter based tracker: `KalmanTracker`.

One can implement their own tracker by extending `BaseTracker` class defined [here](tracking/base_tracker.py)


## Implementation Methodology

The class `DetectAndTrack`, defined [here](pipeline/pipeline.py), maintains list of currently tracked objects.

- Process current frame to obtain new detections
- Assign current detections to existing trackers using Hungarian Algorithm. This would result in matches, unmatched
detections and unmatched trackers
- Assign new trackers to unmatched detections
- Keep old trackers for consecutive unmatched detections for `max_age` frames
- Update tracker's state using tracking algorithm (currently Kalman Filter)   
