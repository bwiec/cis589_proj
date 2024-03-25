#! /usr/bin/env python3

from common.camera import camera
from algorithm.algorithm_edge import algorithm_edge
import sys
import argparse
import cv2
import time

def process(frame):
    print('Processing frame')
    start = time.time()

    end = time.time()
    return end-start

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog=sys.argv[0], description='Edge-based ML implementation on AMD/Xilinx KV260')
    arg_parser.add_argument('--print_duration', action='store_true', help='Print processing duration to console')
    arg_parser.add_argument('--display', action='store_true', help='Display resulting image')
    arg_parser.add_argument('--test_image', action='store', nargs=1, help='Test with an image (i.e. .jpg) instead of camera stream')
    Args = sys.argv
    Args.pop(0)
    args = arg_parser.parse_args(Args)
        
    if args.test_image is not None:
        frame = cv2.imread(args.test_image[0])
        algorithm = algorithm_edge()
        _, process_duration = algorithm.process(frame)
        if args.print_duration:
            print('process_duration: ' + str(process_duration))
        if args.display:
            cv2.imshow('Test image', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        exit()

    cam = camera()
    algorithm = algorithm_edge()
    
    while True:
        frame, capture_duration = cam.read()
        image, process_duration = algorithm.process(frame)
        if args.print_duration:
            print('capture_duration: ' + str(capture_duration))
            print('process_duration: ' + str(process_duration))
        if args.display:
            cam.display(image)
            if cv2.waitKey(1) == ord('q'):
                break
