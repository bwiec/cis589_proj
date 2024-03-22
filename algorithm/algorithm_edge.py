import cv2
import time

class algorithm_edge:
    _print = ''
    _algorithm = ''

    def __init__(self, print=True):
        self._print = print

    def process(self, frame):
        duration = -1
        if self._algorithm == 'detect_faces':
            #duration, faces_count = self._show_faces(frame)
            if self._print:
                print("Faces detected: " + str(faces_count))

        return duration
