#! /usr/bin/env python3
from common.camera import camera
import sys
import argparse
import cv2
import boto3
import time
import io
from PIL import Image, ImageDraw

def process(frame, algorithm):
    print('Processing frame')
    duration = 0
    if algorithm == 'detect_faces':
        start = time.time()
        faces_count = show_faces(frame)
        end = time.time()
        duration = end - start
        print("Faces detected: " + str(faces_count))

    if algorithm == 'detect_labels':
        start = time.time()
        label_count = detect_labels_local_file(frame)
        end = time.time()
        duration = end - start
        print("Labels detected: " + str(label_count))

    return duration

def show_faces(photo):
    client = boto3.client('rekognition')
    with open(photo, 'rb') as image:
        start = time.time()
        response = client.detect_faces(Image={'Bytes': image.read()})
        end = time.time()
        print('inference time: ' + str(end-start))

    image = Image.open(photo)
    
    imgWidth, imgHeight = image.size
    draw = ImageDraw.Draw(image)
    
    # calculate and display bounding boxes for each detected face
    print('Detected faces for ' + photo)
    for faceDetail in response['FaceDetails']:
        #print('The detected face is between ' + str(faceDetail['AgeRange'] ['Low']) + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')
        box = faceDetail['BoundingBox']
        left = imgWidth * box['Left']
        top = imgHeight * box['Top']
        width = imgWidth * box['Width']
        height = imgHeight * box['Height']
        print('Left: ' + '{0:.0f}'.format(left))
        print('Top: ' + '{0:.0f}'.format(top))
        print('Face Width: ' + "{0:.0f}".format(width))
        print('Face Height: ' + "{0:.0f}".format(height))
        points = (
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
            (left, top)
        )
        draw.line(points, fill='#00d400', width=2)
        # Alternatively can draw rectangle. However you can't set line width.
        # draw.rectangle([left,top, left + width, top + height],outline='#00d400')
    image.show()
    return len(response['FaceDetails'])

def detect_labels_local_file(photo):
    client = boto3.client('rekognition')
    
    with open(photo, 'rb') as image:
        start = time.time()
        response = client.detect_labels(Image={'Bytes': image.read()})
        end = time.time()
        print('inference time: ' + str(end-start))
        
    print('Detected labels in ' + photo)
    for label in response['Labels']:
        print(label['Name'] + ' : ' + str(label['Confidence']))
        
    return len(response['Labels'])

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
        process_duration = process(frame, args.algorithm[0])
        if args.print_duration:
            print('process_duration: ' + str(process_duration))
        if args.display:
            cv2.imshow('Test image', cv2.imread(frame))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        exit()

    cam = camera()

    while True:
        frame, capture_duration = cam.read()
        process_duration = process(frame, args.algorithm[0])
        if args.print_duration:
            print('capture_duration: ' + str(capture_duration))
            print('process_duration: ' + str(process_duration))
        if args.display:
            cam.display(frame)
            if cv2.waitKey(1) == ord('q'):
                break
