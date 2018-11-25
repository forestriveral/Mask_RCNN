
from concrete_mrcnn.concrete import *


def train_strategy(command, weights, dataset_path, version, stage, logs_path,
                   limit=None, detect_target=None,
                   filename=None, config_display=True,
                   random_image=False, validate_type="bbox"):

    if command == "detect":
        assert detect_target in ["subset", "images"], \
            "Provide --subset or one image to run prediction on"
        if detect_target == "images":
            assert filename, "Provide --which image do you want to detect?"

    print("Command Executed: ", command)
    print("Weights: ", weights)
    print("Dataset: ", dataset_path)
    if detect_target:
        print("Subset: ", detect_target)
    print("Logs: ", logs_path)
    if command == "train":
        assert stage in ["1", "2", "3"]
        if stage == "1":
            print("==> Train mode: Network heads")
        elif stage == "2":
            print("==> Train mode: Fine tune Resnet stage 4 and up")
        elif stage == "3":
            print("==> Train mode: Fine tune all layers")
        else:
            print("Invalid Train mode. Use the default mode")
            stage = "1"

    # Configurations
    if args.command == "train":
        concrete_config = ConcreteConfig()
    else:
        concrete_config = InferenceConfig()
    if config_display:
        concrete_config.display()

    # Create model
    if command == "train":
        concrete_model = modellib.MaskRCNN(mode="training",
                                           config=concrete_config,
                                           model_dir=logs_path)
    else:
        concrete_model = modellib.MaskRCNN(mode="inference",
                                           config=concrete_config,
                                           model_dir=logs_path)

    # Select weights file to load
    if weights.lower() == "coco":
        weights_dir = COCO_WEIGHTS_PATH
    elif weights.lower() == "last":
        # Find last trained weights
        weights_dir = concrete_model.find_last()
    # elif args.model.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     model_path = model.get_imagenet_weights()
    else:
        weights_dir = weights

    # Load weights
    print("Loading weights ", weights_dir)
    if weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        concrete_model.load_weights(weights_dir, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        concrete_model.load_weights(weights_dir, by_name=True)

    # Train or evaluate or detect
    if command == "train":
        train(concrete_model, dataset_path, command, version, stage)
    elif command == "val":
        # Validation dataset
        val_dataset = ConcreteDataset()
        coco_data = dataset_val.load_concrete(dataset_path, command, version, return_coco=True)
        val_dataset.prepare()
        print("Running concrete evaluation on {} images.".format(limit))
        if limit == 'all':
            limit = len(val_dataset.image_ids)
        evaluate_concrete(concrete_model, val_dataset, coco_data,
                          validate_type, limit=int(limit))
    elif command == "detect":
        if detect_target == "subset":
            detect(concrete_model, dataset_path, "test", version)
        else:
            # spcific class name depends on datasets used
            classes = ['BG', 'bughole']
            files = next(os.walk(IMAGE_DIR))[2]
            if random_image:
                file_name = random.choice(files)
            else:
                file_name = filename
            pic = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

            # Run detection
            result = concrete_model.detect([pic], verbose=1)

            # Visualize results
            res = result[0]
            visualize.display_instances(pic, res['rois'], res['masks'], res['class_ids'],
                                        classes, res['scores'])
    else:
        print("'{}' is not recognized. "
              "Use 'train', 'val' or 'detect'".format(command))
