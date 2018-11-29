import argparse
import logging
import time

from moviepy.editor import VideoFileClip

from pipeline.pipeline import DetectAndTrack
from tracking.kalman_tracker import KalmanTracker

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect and Track')
    parser.add_argument("--input", dest="input_file",
                        help="path for video file",
                        default="video.mp4")
    parser.add_argument("--output", dest="output_file",
                        help="path for processed video file",
                        default="output.mp4")
    parser.add_argument("--detector", dest='detector',
                        help="mobilenet or yolov3",
                        default="yolov3", type=str)
    parsed_args = parser.parse_args()

    assert parsed_args.detector.lower() in ["mobilenet", "yolov3"]

    if parsed_args.detector.lower() == "mobilenet":
        from detection.mobilenet_ssd_detector import Detector
        detector = Detector()
    else:
        from detection.yolo_v3_detector import Detector
        detector = Detector()

    detect_and_track = DetectAndTrack(detector, KalmanTracker)
    start = time.time()
    clip1 = VideoFileClip(parsed_args.input_file)
    clip = clip1.fl_image(detect_and_track.pipeline)
    clip.write_videofile(parsed_args.output_file, audio=False)
    end = time.time()
    print(round(end - start, 2), 'Seconds to finish')
