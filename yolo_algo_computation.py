from yolo_utils import scale_boxes
from yad2k.models.keras_yolo import yolo_boxes_to_corners
import numpy as np

import PIL
import tensorflow as tf
from keras import backend as K

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    box_scores =  box_confidence * box_class_probs
    box_classes = K.argmax(box_scores,axis=-1)
    box_class_scores = K.max(box_scores,axis=-1)
    filtering_mask = (box_class_scores >= threshold)
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    return scores, boxes, classes
	
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) 
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes_tensor)
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)
    
    return scores, boxes, classes
	
def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    
	box_xy, box_wh,box_confidence, box_class_probs = yolo_outputs
	boxes = yolo_boxes_to_corners(box_xy, box_wh)
	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
	boxes = scale_boxes(boxes, image_shape)
	boxes_before_nms = (scores, boxes, classes)
	scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

	return scores, boxes, classes ,boxes_before_nms
	
