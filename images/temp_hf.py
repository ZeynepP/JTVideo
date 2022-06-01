import itertools

import torch
import torchvision
from PIL import Image
import numpy as np
import requests
# COCO classes
from transformers import DetrFeatureExtractor, DetrForObjectDetection

video_frames = torchvision.io.read_video("../0/7/201700001.mp4")



CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
url = './data/000000039769.jpg'
image = Image.open(url)

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')


def get_objects(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    np_softmax = (logits.softmax(-1)[0, :, :-1]).detach().numpy()
    threshold = 0.7
    for i, j in enumerate(np_softmax):
        if np.max(j) > threshold:
            print(CLASSES[np.argmax(j)])

    print("over over")

video_frames = torchvision.io.read_video("../0/7/201700001.mp4",start_pts=10,pts_unit="sec" )
counter = 0
for im in video_frames[0]:
    get_objects(im)
    torchvision.io.write_jpeg(im, "./data/" + str(counter) + ".jpg")
    counter += 1
