

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import json
import sys
import random
# import math
# import re
import time
import numpy as np
import cv2
import imgaug
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# %matplotlib inline
DATASETS_PATH = "datasets"

# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "output/concrete/")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, DATASETS_PATH)


############################################################
#  Configurations
############################################################

class ConcreteConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "concrete"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 2  # background + 3 shapes
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    IMAGE_MIN_SCALE = 0

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 25

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

# config = ConcreteConfig()
# config.display()

class InferenceConfig(ConcreteConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MIN_CONFIDENCE = 0.7

    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

############################################################
#  Dataset
############################################################

class ConcreteDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_concrete(self, dataset_dir, subset, version, return_coco=False,
                      dataset_name="concrete"):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """

        # Train or validation dataset?
        assert subset in ["train", "val", "test"], "Invalid subset"
        if subset == "train" or "val":
            coco = COCO("{}/annotations/{}_{}{}.json".format(dataset_dir, dataset_name, subset, version))
            image_dir = "{}/{}{}".format(dataset_dir, subset, version)

            # Get class and images ids
            class_ids = sorted(coco.getCatIds())
            image_ids = list(coco.imgs.keys())

            # Add classes
            for i in class_ids:
                self.add_class("concrete", i, coco.loadCats(i)[0]["name"])

            # Add images
            for i in image_ids:
                self.add_image(
                    "concrete", image_id=i,
                    path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                    width=coco.imgs[i]["width"],
                    height=coco.imgs[i]["height"],
                    annotations=coco.loadAnns(coco.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)))
            if return_coco:
                return coco
        else:
            dataset_path = os.path.join(dataset_dir, subset + version)
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_path))[2]

            # Add images
            for image_id in image_ids:
                self.add_image(
                    "concrete",
                    image_id=image_id,
                    path=os.path.join(dataset_path, "{}.png".format(image_id)))

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        # print(info)
        if info["source"] == "concrete":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "concrete":
            return super(ConcreteDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # print(annotations)
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            # print(annotation)
            class_id = self.map_source_class_id(
                "concrete.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(ConcreteDataset, self).load_mask(image_id)

            # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle concrete_mrcnn
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

############################################################
#  Training
############################################################

def train(model, dataset_dir, subset, version, train_mode):
    """Train the model."""
    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = ConcreteDataset()
    dataset_train.load_concrete(dataset_dir, subset, version)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ConcreteDataset()
    dataset_val.load_concrete(dataset_dir, subset, version)
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    # augmentation = imgaug.augmenters.Fliplr(0.5)
    augmentation = imgaug.augmenters.SomeOf((0, 2), [
        imgaug.augmenters.Fliplr(0.5),
        imgaug.augmenters.Flipud(0.3),
        imgaug.augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),
                                                per_channel=0.5),
        imgaug.augmenters.CropAndPad(percent=(-0.05, 0.1), pad_mode=imgaug.ALL, pad_cval=(0, 255)),
        imgaug.augmenters.GaussianBlur(sigma=(0.0, 3.0)),
        imgaug.augmenters.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
        imgaug.augmenters.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)
        ], random_order=True)

    # *** This training schedule is an example. Update to your needs ***

    if train_mode == "1":
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers='heads',
                    augmentation=augmentation)

    elif train_mode == "2":
        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=90,
                    layers='4+',
                    augmentation=augmentation)

    elif train_mode == "3":
        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=120,
                    layers='all',
                    augmentation=augmentation)
    else:
        print("'{}' is Invalid training mode".format(args.train_mode))


############################################################
#  Evaluation
############################################################

def build_concrete_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_concrete(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    concrete_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_concrete_results(dataset, concrete_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    concrete_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, concrete_results, eval_type)
    cocoEval.params.imgIds = concrete_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset, version):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = ConcreteDataset()
    dataset.load_concrete(dataset_dir, subset)
    dataset.prepare()

    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")
        plt.savefig("{}/{}".format(submit_dir,
                                   dataset.image_info[image_id]["path"].split('/')[-1]))

    # Save to CSV file
    submission = "ImageId, EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Run Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Concrete COCO-datasets.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' or 'detect' on Concrete COCO-datasets")
    parser.add_argument('--weights', required=False,
                        default="coco",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--dataset', required=False,
                        default=DATASETS_PATH,
                        metavar="/path/to/coco/",
                        help='Directory of the Concrete COCO-datasets dataset')
    parser.add_argument('--train_mode', required=False,
                        default= "1",
                        metavar="Training - Stage 1 ,2 or 3",
                        help="Choose to train in different stages: \n" +
                             "Stage 1 ==> Training network heads (default) \n" +
                             "Stage 2 ==> Fine tune Resnet stage 4 and up \n" +
                             "Stage 3 ==> Fine tune all layers \n")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default= 20,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=20)')
    parser.add_argument('--subset', required=False,
                        metavar="Specific dataset sub-directory to detection",
                        help="Subset of dataset to run test")
    parser.add_argument('--filename', required=False,
                        metavar="Choose specific images you want to be detected ",
                        help="Image filename provided when detect one images")
    args = parser.parse_args()

    if args.command == "detect":
        assert args.subset, "Provide --subset or one image to run prediction on"
        if args.subset == "images":
            assert args.filename, "Provide --which image do you want to detect?"

    print("Command: ", args.command)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)
    if args.command == "train":
        assert args.train_mode in ["1", "2", "3"]
        if args.train_mode =="1":
            print("Train mode: Network heads")
        elif args.train_mode =="2":
            print("Train mode: Fine tune Resnet stage 4 and up")
        elif args.train_mode =="3":
            print("Train mode: Fine tune all layers")
        else:
            print("Invalid Train mode. Use the default mode")
            args.train_mode = "1"


    # Configurations
    if args.command == "train":
        config = ConcreteConfig()
    else:
        config = InferenceConfig()
    config.display()
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    # elif args.model.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     model_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate or detect
    if args.command == "train":
        train(model, args.dataset, "train", args.train_mode)
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = ConcreteDataset()
        val_type = "val"
        coco = dataset_val.load_concrete(args.dataset, val_type, return_coco=True)
        dataset_val.prepare()
        print("Running concrete evaluation on {} images.".format(args.limit))
        evaluate_concrete(model, dataset_val, coco, "bbox", limit=int(args.limit))
    elif args.command == "detect":
        assert args.subset in ["test","images"]
        if args.subset == "test":
            detect(model, args.dataset, "test")
        else:
            class_names = ['BG', 'bughole']
            file_names = next(os.walk(IMAGE_DIR))[2]
            image = skimage.io.imread(os.path.join(IMAGE_DIR, args.filename))

            # Run detection
            results = model.detect([image], verbose=1)

            # Visualize results
            r = results[0]
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'])
    else:
        print("'{}' is not recognized. "
              "Use 'train', 'evaluate' or 'detect'".format(args.command))

