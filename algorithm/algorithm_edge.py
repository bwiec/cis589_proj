import cv2
import time
from pynq_dpu import DpuOverlay
import os
import numpy as np
import random
import colorsys
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

class algorithm_edge:
    _print = ''
    _overlay = ''
    _algorithm = ''
    _anchors = ''
    _classes_path = "algorithm/voc_classes.txt"
    _num_classes = 0
    _colors = ''
    _dpu = ''
    _shapeIn = ''
    _shapeOut0 = ''
    _shapeOut1 = ''
    _shapeOut2 = ''
    _input_data = ''
    _output_data = ''
    _image = ''   

    def __init__(self, print=True):
        self._print = print
        self._overlay = DpuOverlay("dpu.bit")
        self._overlay.load_model("tf_yolov3_voc.xmodel")
        
        anchor_list = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
        anchor_float = [float(x) for x in anchor_list]
        self._anchors = np.array(anchor_float).reshape(-1, 2)
        
        self._class_names = self._get_class(self._classes_path)
        
        self._num_classes = len(self._class_names)
        hsv_tuples = [(1.0 * x / self._num_classes, 1., 1.) for x in range(self._num_classes)]
        self._colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self._colors = list(map(lambda x: 
                          (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), 
                          self._colors))
        random.seed(0)
        random.shuffle(self._colors)
        random.seed(None)
        
    def _get_class(self, classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
        
    def process(self, frame):
        duration = -1
        image, duration = self._run(frame)
        return image, duration

    def _run(self, frame):
        self._dpu = self._overlay.runner
        inputTensors = self._dpu.get_input_tensors()
        outputTensors = self._dpu.get_output_tensors()
        self._shapeIn = tuple(inputTensors[0].dims)
        
        self._shapeOut0 = (tuple(outputTensors[0].dims)) # (1, 13, 13, 75)
        self._shapeOut1 = (tuple(outputTensors[1].dims)) # (1, 26, 26, 75)
        self._shapeOut2 = (tuple(outputTensors[2].dims)) # (1, 52, 52, 75)
        
        outputSize0 = int(outputTensors[0].get_data_size() / self._shapeIn[0]) # 12675
        outputSize1 = int(outputTensors[1].get_data_size() / self._shapeIn[0]) # 50700
        outputSize2 = int(outputTensors[2].get_data_size() / self._shapeIn[0]) # 202800
        
        self._input_data = [np.empty(self._shapeIn, dtype=np.float32, order="C")]
        self._output_data = [np.empty(self._shapeOut0, dtype=np.float32, order="C"), 
                       np.empty(self._shapeOut1, dtype=np.float32, order="C"),
                       np.empty(self._shapeOut2, dtype=np.float32, order="C")]
        self._image = self._input_data[0]
        
        image, duration = self._run_dpu(frame, display=True)
        return image, duration
        
    def _run_dpu(self, frame, display=False):
        input_image = frame
    
        # Pre-processing
        image_size = input_image.shape[:2]
        image_data = np.array(self._pre_process(input_image, (416, 416)), dtype=np.float32)
        
        # Fetch data to DPU and trigger it
        self._image[0,...] = image_data.reshape(self._shapeIn[1:])
        start = time.time()
        job_id = self._dpu.execute_async(self._input_data, self._output_data)
        self._dpu.wait(job_id)
        end = time.time()
        
        # Retrieve output data
        conv_out0 = np.reshape(self._output_data[0], self._shapeOut0)
        conv_out1 = np.reshape(self._output_data[1], self._shapeOut1)
        conv_out2 = np.reshape(self._output_data[2], self._shapeOut2)
        yolo_outputs = [conv_out0, conv_out1, conv_out2]
        
        # Decode output from YOLOv3
        boxes, scores, classes = self._evaluate(yolo_outputs, image_size, self._class_names, self._anchors)
        
        if display:
            #_ = self._draw_boxes(input_image, boxes, scores, classes)
            image = self._draw_bbox(input_image, boxes, classes)

        print("Number of detected objects: {}".format(len(boxes)))
        return image, end-start
        
    def _letterbox_image(self, image, size):
        ih, iw, _ = image.shape
        w, h = size
        scale = min(w/iw, h/ih)

        nw = int(iw*scale)
        nh = int(ih*scale)

        image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
        new_image = np.ones((h,w,3), np.uint8) * 128
        h_start = (h-nh)//2
        w_start = (w-nw)//2
        new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
        return new_image

    def _pre_process(self, image, model_image_size):
        image = image[...,::-1]
        image_h, image_w, _ = image.shape

        if model_image_size != (None, None):
            assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = self._letterbox_image(image, tuple(reversed(model_image_size)))
        else:
            new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
            boxed_image = self._letterbox_image(image, new_image_size)
        self._image_data = np.array(boxed_image, dtype='float32')
        self._image_data /= 255.
        self._image_data = np.expand_dims(self._image_data, 0) 	
        return self._image_data
        
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)
        anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
        grid_size = np.shape(feats)[1:3]
        nu = num_classes + 5
        predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
        grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = np.concatenate([grid_x, grid_y], axis = -1)
        grid = np.array(grid, dtype=np.float32)

        box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
        box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
        box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
        box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
        return box_xy, box_wh, box_confidence, box_class_probs


    def _correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape, dtype = np.float32)
        image_shape = np.array(image_shape, dtype = np.float32)
        new_shape = np.around(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis = -1)
        boxes *= np.concatenate([image_shape, image_shape], axis = -1)
        return boxes

    def _boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        boxes = self._correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = np.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = np.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores
        
    def _draw_bbox(self, image, bboxes, classes):
        """
        bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        """
        num_classes = len(classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            fontScale = 0.5
            #score = bbox[4]
            #class_ind = int(bbox[5])
            #class_ind = int(0)
            #bbox_color = colors[class_ind]
            bbox_color = tuple([color/1 for color in self._colors[classes[i]]])
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            #c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        return image


    def _nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2-x1+1)*(y2-y1+1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= 0.55)[0]  # threshold
            order = order[inds + 1]

        return keep
        
    def _draw_boxes(self, image, boxes, scores, classes):
        _, ax = plt.subplots(1)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_h, image_w, _ = image.shape

        for i, bbox in enumerate(boxes):
            [top, left, bottom, right] = bbox
            width, height = right - left, bottom - top
            center_x, center_y = left + width*0.5, top + height*0.5
            score, class_index = scores[i], classes[i]
            label = '{}: {:.4f}'.format(self._class_names[class_index], score) 
            color = tuple([color/255 for color in self._colors[class_index]])
            ax.add_patch(Rectangle((left, top), width, height,
                                   edgecolor=color, facecolor='none'))
            ax.annotate(label, (center_x, center_y), color=color, weight='bold', 
                        fontsize=12, ha='center', va='center')

    def _evaluate(self, yolo_outputs, image_shape, class_names, anchors):
        score_thresh = 0.2
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = np.shape(yolo_outputs[0])[1 : 3]
        input_shape = np.array(input_shape)*32

        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self._boxes_and_scores(
                yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), 
                input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = np.concatenate(boxes, axis = 0)
        box_scores = np.concatenate(box_scores, axis = 0)

        mask = box_scores >= score_thresh
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(len(class_names)):
            class_boxes_np = boxes[mask[:, c]]
            class_box_scores_np = box_scores[:, c]
            class_box_scores_np = class_box_scores_np[mask[:, c]]
            nms_index_np = self._nms_boxes(class_boxes_np, class_box_scores_np) 
            class_boxes_np = class_boxes_np[nms_index_np]
            class_box_scores_np = class_box_scores_np[nms_index_np]
            classes_np = np.ones_like(class_box_scores_np, dtype = np.int32) * c
            boxes_.append(class_boxes_np)
            scores_.append(class_box_scores_np)
            classes_.append(classes_np)
        boxes_ = np.concatenate(boxes_, axis = 0)
        scores_ = np.concatenate(scores_, axis = 0)
        classes_ = np.concatenate(classes_, axis = 0)

        return boxes_, scores_, classes_

    def __del__(self):
        del self._overlay
        del self._dpu