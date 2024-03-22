#! /usr/bin/env python3
from common.camera import camera
from algorithm.algorithm_cloud import algorithm_cloud
import sys
import argparse
import cv2

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog=sys.argv[0], description='Edge-based ML implementation on AMD/Xilinx KV260')
    arg_parser.add_argument('--print_duration', action='store_true', help='Print processing duration to console')
    arg_parser.add_argument('--display', action='store_true', help='Display resulting image')
    arg_parser.add_argument('--test_image', action='store', nargs=1, help='Test with an image (i.e. .jpg) instead of camera stream')
    arg_parser.add_argument('--algorithm', action='store', default='detect_faces', nargs=1, help='Choose which algorithm to run (detect_faces, detect_labels)')
    Args = sys.argv
    Args.pop(0)
    args = arg_parser.parse_args(Args)

    if args.algorithm[0] != 'detect_faces' and args.algorithm[0] != 'detect_labels':
        raise Exception("Unsupported algorithm " + str(args.algorithm))
    
    if args.test_image != '':
        frame = args.test_image[0]
        algorithm = algorithm_cloud(args.algorithm[0])
        process_duration = algorithm.process(frame)
        if args.print_duration:
            print('process_duration: ' + str(process_duration))
        if args.display:
            cv2.imshow('Test image', cv2.imread(frame))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        exit()

    cam = camera()

    algorithm = algorithm_cloud(args.algorithm[0])
    while True:
        frame, capture_duration = cam.read()
        process_duration = algorithm.process(frame)
        if args.print_duration:
            print('capture_duration: ' + str(capture_duration))
            print('process_duration: ' + str(process_duration))
        if args.display:
            cam.display(frame)
            if cv2.waitKey(1) == ord('q'):
                break
