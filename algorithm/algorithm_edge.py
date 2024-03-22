import cv2
import time

class algorithm_edge:
    _print = ''
    _algorithm = ''

    def __init__(self, print=True, algorithm='detect_faces'):
        self._print = print
        self._algorithm = algorithm

    def process(self, frame):
        duration = -1
        if self._algorithm == 'detect_faces':
            duration, faces_count = self._show_faces(frame)
            if print:
                print("Faces detected: " + str(faces_count))

        return duration
