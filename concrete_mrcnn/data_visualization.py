

from concrete import *


# Training dataset
dataset_train = ConcreteDataset()
dataset_train.load_concrete()
dataset_train.prepare()

# map = dataset_train.class_from_source_map
# print(map)

# Validation dataset
# dataset_val = ConcreteDataset()
# dataset_val.load_shapes(subset="val")
# dataset_val.prepare()

# class information of datasets
# print("Image Count: {}".format(len(dataset_train.image_ids)))
# print("Class Count: {}".format(dataset_train.num_classes))
# for i, info in enumerate(dataset_train.class_info):
#     print("{:3}. {:50}".format(i, info['name']))

# ============================================== Sample Display
# image_ids = np.random.choice(dataset_train.image_ids, 2)
# # print(image_ids)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# ==============================================  Bounding Boxes
# Load random image and mask.
# image_id = np.random.choice(dataset_train.image_ids)
# image = dataset_train.load_image(image_id)
# mask, class_ids = dataset_train.load_mask(image_id)
# # Compute Bounding box
# bbox = utils.extract_bboxes(mask)
#
# # Display image and additional stats
# print("image_id ", image_id, dataset_train.image_reference(image_id))
# log("image", image)
# log("mask", mask)
# log("class_ids", class_ids)
# log("bbox", bbox)
# # Display image and instances
# visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)

# ======================================== Resized Images

# Load random image and mask.
# image_id = np.random.choice(dataset_train.image_ids, 1)[0]
# image = dataset_train.load_image(image_id)
# mask, class_ids = dataset_train.load_mask(image_id)
# original_shape = image.shape
# # Resize
# image, window, scale, padding, _ = utils.resize_image(
#     image,
#     min_dim=config.IMAGE_MIN_DIM,
#     max_dim=config.IMAGE_MAX_DIM,
#     mode=config.IMAGE_RESIZE_MODE)
# mask = utils.resize_mask(mask, scale, padding)
# # Compute Bounding box
# bbox = utils.extract_bboxes(mask)
#
# # Display image and additional stats
# print("image_id: ", image_id, dataset_train.image_reference(image_id))
# print("Original shape: ", original_shape)
# log("image", image)
# log("mask", mask)
# log("class_ids", class_ids)
# log("bbox", bbox)
# # Display image and instances
# visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)


# ============================================ Unresized Mask

# image_id = np.random.choice(dataset_train.image_ids, 1)[0]
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#     dataset_train, config, image_id, use_mini_mask=False)
#
# log("image", image)
# log("image_meta", image_meta)
# log("class_ids", class_ids)
# log("bbox", bbox)
# log("mask", mask)
#
# visualize.display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
# visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)

# ============================================ Resize Mask (Mini Mask)

# # Add augmentation and mask resizing.
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#     dataset_train, config, image_id, augmentation=True, use_mini_mask=True)
# log("mask", mask)
# visualize.display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
#
# mask = utils.expand_mask(bbox, mask, image.shape)
# visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)

# ============================================ Training perparation

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')





