#! /usr/bin/env python3

from common.camera import camera
import cv2
import time

def process(frame):
    print('Processing frame')
    start = time.time()

    end = time.time()
    return end-start

if __name__ == "__main__":
    cam = camera()

    while True:
        frame, capture_duration = cam.read()
        process_duration = process(frame)
        print('capture_duration: ' + str(capture_duration))
        print('process_duration: ' + str(process_duration))
        cam.display(frame)
        if cv2.waitKey(1) == ord('q'):
            break
