import time

from moviepy.editor import VideoFileClip

from tracking.pipeline import DetectAndTrack
from tracking.detector.MobileNetDetector import CarDetector
import logging

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    detector = CarDetector()
    detect_and_track = DetectAndTrack(detector)
    start = time.time()
    output = 'test_output.mp4'
    clip1 = VideoFileClip("car25_compressed.mp4").subclip(3 * 60)  # The first 8 seconds doesn't have any cars...
    clip = clip1.fl_image(detect_and_track.pipeline)
    clip.write_videofile(output, audio=False)
    end = time.time()
    print(round(end - start, 2), 'Seconds to finish')
