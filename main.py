import logging
import time

from moviepy.editor import VideoFileClip
# from detection.mobilenet_ssd_detector import Detector
from detection.yolo_v3_detector import Detector


from pipeline.pipeline import DetectAndTrack

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    detector = Detector()
    detect_and_track = DetectAndTrack(detector)
    start = time.time()
    output = 'test_output.mp4'
    clip1 = VideoFileClip("car25_compressed.mp4").subclip(180, 210)
    clip = clip1.fl_image(detect_and_track.pipeline)
    clip.write_videofile(output, audio=False)
    end = time.time()
    print(round(end - start, 2), 'Seconds to finish')
