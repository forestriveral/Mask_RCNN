
from concrete_mrcnn.concrete_tools import *
import os
import sys
# sys.path.append("../")

# train_strategy(command="train", weights="last", config_display=False,
#                dataset=DATASETS_PATH, version='000', logs=DEFAULT_LOGS_DIR,
#                stage="3", train_epochs=25,  # train
#                dir_suffix=None,  # specific target directory
#                limit=None, validate_type=None,  # val
#                save=False, csv=False, val_subset=None,  # detect multiple images and save
#                detect_target=None, classes_name=None,  # detect single image
#                image_name=None, random_image=False)

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference",
                          model_dir=DEFAULT_LOGS_DIR,
                          config=config)

image_path = "test000/images/20.jpg"
weight_path = "concrete20181128T2034/mask_rcnn_concrete_0022.h5"

model.load_weights(os.path.join(DEFAULT_LOGS_DIR, weight_path), by_name=True)
class_names = ['BG', "crack", "bughole"]

image = skimage.io.imread(os.path.join(DATASETS_PATH, image_path))
results = model.detect([image], verbose=1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'], figsize=(10, 10))

