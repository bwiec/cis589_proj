import boto3
import time
import io
import cv2
import base64
from PIL import Image, ImageDraw
import numpy as np

class algorithm_cloud:
    _print = ''
    _algorithm = ''
    _client = ''

    def __init__(self, print=True, algorithm='detect_faces'):
        self._print = print
        self._algorithm = algorithm
        self._client = boto3.client('rekognition')

    def process(self, frame):
        duration = -1
        if self._algorithm == 'detect_faces':
            image, duration, faces_count = self._show_faces(frame)
            if self._print:
                print("Faces detected: " + str(faces_count))

        if self._algorithm == 'detect_labels':
            image, duration, label_count = self._detect_labels_local_file(frame)
            if self._print:
                print("Labels detected: " + str(label_count))

        return image, duration

    def _show_faces(self, photo):
        duration = -1
        start = time.time()
        ret, buffer = cv2.imencode('.jpg', photo)
        image_data = base64.b64encode(buffer)
        image_data_binary = base64.decodebytes(image_data)
        response = self._client.detect_faces(Image={'Bytes': image_data_binary})
        end = time.time()
        duration = end-start
        if self._print:
            print('inference time: ' + str(duration))

        #image = Image.open(photo)
        image = Image.fromarray(photo)
        
        imgWidth, imgHeight = image.size
        draw = ImageDraw.Draw(image)
        
        # calculate and display bounding boxes for each detected face
        if self._print:
            print('Detected faces for camera stream')
        for faceDetail in response['FaceDetails']:
            #print('The detected face is between ' + str(faceDetail['AgeRange'] ['Low']) + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')
            box = faceDetail['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']
            if self._print:
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
        #image.show()
        image_np = np.asarray(image)
        return image_np, duration, len(response['FaceDetails'])

    def _detect_labels_local_file(self, photo):
        duration = -1
        start = time.time()
        ret, buffer = cv2.imencode('.jpg', photo)
        image_data = base64.b64encode(buffer)
        image_data_binary = base64.decodebytes(image_data)
        response = self._client.detect_labels(Image={'Bytes': image_data_binary})
        end = time.time()
        duration = end-start
        if self._print:
            print('inference time: ' + str(duration))
            
            
            
        image = Image.fromarray(photo)
        
        imgWidth, imgHeight = image.size
        draw = ImageDraw.Draw(image)
        
        # calculate and display bounding boxes for each detected face
        if self._print:
            print('Detected labels for camera stream')
        for Labels in response['Labels']:
            for Instances in Labels['Instances']:
                box = Instances['BoundingBox']
                left = imgWidth * box['Left']
                top = imgHeight * box['Top']
                width = imgWidth * box['Width']
                height = imgHeight * box['Height']
                if self._print:
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
            #image.show()
        image_np = np.asarray(image)
          
        
        if self._print:
            print('Detected labels in caamera stream')
            for label in response['Labels']:
                print(label['Name'] + ' : ' + str(label['Confidence']))
            
        return image_np, duration, len(response['Labels'])
