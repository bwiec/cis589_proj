#! /usr/bin/env python3
import cv2
import boto3

def detect_labels_local_file(photo):
    client = boto3.client('rekognition')
    
    with open(photo, 'rb') as image:
        response = client.detect_labels(Image={'Bytes': image.read()})
        
    print('Detected labels in ' + photo)
    for label in response['Labels']:
        print(label['Name'] + ' : ' + str(label['Confidence']))
        
    return len(response['Labels'])

photo = 'sample.jpg'
label_count = detect_labels_local_file(photo)
print("Labels detected: " + str(label_count))

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