
from concrete_mrcnn.concrete import *


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
            # file_names = next(os.walk(IMAGE_DIR))[2]
            # image = skimage.io.imread(os.path.join(IMAGE_DIR, args.filename))

            # Run detection
            # results = model.detect([image], verbose=1)

            # Visualize results
            # r = results[0]
            # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
            #                             class_names, r['scores'])
    else:
        print("'{}' is not recognized. "
              "Use 'train', 'evaluate' or 'detect'".format(args.command))