# detects blink and POSTS endpoint with data
import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

import RPi.GPIO as GPIO
import time as time
from time import sleep

import adafruit_vl6180x

blinkPerFrame = []

WORKSPACE_PATH = '../Tensorflow/workspace'
SCRIPTS_PATH = '../Tensorflow/scripts'
APIMODEL_PATH = '../Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-5')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

#ret: blinks per second
def detectBlink(blinkPerFrame, framerate):
    FRAME_GAP = 120
	if len(blinkPerFrame) < FRAME_GAP:
        return 0
	else:
		total = 0
		for i in range(FRAME_GAP):
			total = total + blinkPerFrame(len(blinkPerFrame) - i)
		blinkRate = (total / FRAME_GAP) * framerate
		return blinkRate

#@param blinks per second
def isTired(blinkRate):
    GPIO.setmode(GPIO.BCM)
	ABR = 0.1
    buzzer = 14
    GPIO.setup(buzzer,GPIO.OUT)
    tired = blinkRate>ABR
	if tired:
        for i in range(3)
            GPIO.output(buzzer,GPIO.HIGH)
            sleep(.2)
            GPIO.output(buzzer,GPIO.LOW)
            sleep(.2)
	return tired

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# Setup capture
cap = cv2.VideoCapture(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# lidar
sensor = adafruit_vl6180x.VL6180X(i2c)

while True:

    ret, frame = cap.read()
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
    for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.5,
                agnostic_mode=False)

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

    #lidar stuff
    if sensor.range > 5:
        continue

    # This is the way I'm getting my coordinates
    boxes = detections['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = detections['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh=.5
    # iterate over all objects found
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            print (str(scores[i] * 100) + "%", detections['detection_classes'][i])
            # 1 is blink
            isBlink = 1 if detections['detection_classes'][i] == 0 else 0;
            blinkPerFrame.append(isBlink)
            bps = detectBlink(blinkPerFrame,60)
            isTired(bps)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break