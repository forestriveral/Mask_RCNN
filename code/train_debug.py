

from configuration import *


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

# image_ids = np.random.choice(dataset_train.image_ids, 2)
# # print(image_ids)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Load random image and mask.
image_id = np.random.choice(dataset_train.image_ids)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset_train.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)







