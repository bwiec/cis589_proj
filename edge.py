#! /usr/bin/env python3

from common.camera import camera
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
    Args = sys.argv
    Args.pop(0)
    args = arg_parser.parse_args(Args)

    cam = camera()

    while True:
        frame, capture_duration = cam.read()
        process_duration = process(frame)
        if args.print_duration:
            print('capture_duration: ' + str(capture_duration))
            print('process_duration: ' + str(process_duration))
        if args.display:
            cam.display(frame)
            if cv2.waitKey(1) == ord('q'):
                break
