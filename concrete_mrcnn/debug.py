
# from concrete_mrcnn.concrete_tools import *
import os
import cv2
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concrete_mrcnn.visualization import plot_loss_curve
# sys.path.append("../")

# train_strategy(command="train", weights="last", config_display=False,
#                dataset=DATASETS_PATH, version='000', logs=DEFAULT_LOGS_DIR,
#                stage="3", train_epochs=25,  # train
#                dir_suffix=None,  # specific target directory
#                limit=None, validate_type=None,  # val
#                save=False, csv=False, val_subset=None,  # detect multiple images and save
#                detect_target=None, classes_name=None,  # detect single image
#                image_name=None, random_image=False)

# config = InferenceConfig()
# model = modellib.MaskRCNN(mode="inference",
#                           model_dir=DEFAULT_LOGS_DIR,
#                           config=config)
#
# weight_path = "concrete20181128T2034/mask_rcnn_concrete_0022.h5"
# model.load_weights(os.path.join(DEFAULT_LOGS_DIR, weight_path), by_name=True)
# class_names = ['BG', "crack", "bughole"]
#
# image_path = os.path.join(DATASETS_PATH, "test000/images")
# images = next(os.walk(image_path))[2]
#
# for image in images:
#     image = os.path.join(image_path, image)
#     image = skimage.io.imread(image)
#     results = model.detect([image], verbose=1)
#     r = results[0]
#     visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                                 class_names, r['scores'], figsize=(10, 10))

load_path = "../logs/concrete20181129T1546/training.csv"
plot_loss_curve(load_path)
