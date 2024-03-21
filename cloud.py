#! /usr/bin/env python3
import cv2
import boto3
import time
import io
from PIL import Image, ImageDraw

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


label_count = detect_labels_local_file('labels_sample.jpg')
print("Labels detected: " + str(label_count))

faces_count = show_faces('faces_sample.jpg')
print("Faces detected: " + str(faces_count))
exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()