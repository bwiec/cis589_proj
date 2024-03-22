import cv2
import time

class camera:
    _dev_num = 0
    _hsize = 0
    _vsize = 0
    _frame = 0
    _cap = 0

    def __init__(self, dev_num=0, hsize=640, vsize=480):
        self._dev_num = dev_num
        self._hsize = hsize
        self._vsize = vsize
        self.__open()

    def __open(self):
        self._cap = cv2.VideoCapture(self._dev_num)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._hsize)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._vsize)
        if not self._cap.isOpened():
            raise Exception("Cannot open camera " + str(self._dev_num) + " with " + str(self._hsize) + "x" + str(self._vsize))

    def read(self):
        start = time.time()
        ret, self._frame = self._cap.read()
        end = time.time()
        return self._frame, end-start
    
    def display(self, frame):
        cv2.imshow('Camera', frame)

    def __del__(self):
        self._cap.release()
        cv2.destroyAllWindows()