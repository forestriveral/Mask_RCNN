
import glob
from scipy import interpolate
from concrete_mrcnn.concrete import *
from samples.coco import coco


def train_strategy(command, weights, dataset, version, stage, logs,
                   val_subset=None, dir_suffix=None, limit=None,
                   detect_target=None, classes_name=None, image_name=None,
                   config_display=False, train_epochs=None, random_image=False,
                   validate_type=None, save=False, csv=False):

    if command == "detect":
        assert detect_target in ["subset", "images"], \
            "Provide --subset or one image to run prediction on"
        if detect_target == "images":
            if not random_image:
                assert image_name, "Provide --which image do you want to detect?"

    print("Command Executed: ", command)
    print("Weights: ", weights)
    print("Dataset: ", dataset)
    if detect_target:
        print("Detect target: ", detect_target)
    print("Logs: ", logs)
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
    if command == "train":
        concrete_config = ConcreteConfig()
    else:
        if weights.lower() == "debug":
            concrete_config = CocoConfig()
        else:
            concrete_config = InferenceConfig()
    if config_display:
        concrete_config.display()

    # Create model
    # if True:
    with tf.device("/gpu:1"):
        if command == "train":
            concrete_model = modellib.MaskRCNN(mode="training",
                                               config=concrete_config,
                                               model_dir=logs)
        else:
            concrete_model = modellib.MaskRCNN(mode="inference",
                                               config=concrete_config,
                                               model_dir=logs)

    # Select weights file to load
    if weights.lower() == "coco":
        weights_dir = COCO_WEIGHTS_PATH
    elif weights.lower() == "last":
        # Find last trained weights
        weights_dir = concrete_model.find_last()
        # print("last weight:", concrete_model.find_last())
    elif weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_dir = concrete_model.get_imagenet_weights()
    else:
        weights_dir = weights

    # Load weights
    print("Loading weights ", weights_dir)
    if weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        concrete_model.load_weights(weights_dir, by_name=True,
                                    exclude=["mrcnn_class_logits",
                                             "mrcnn_bbox_fc", "mrcnn_bbox",
                                             "mrcnn_mask"])
    else:
        concrete_model.load_weights(weights_dir, by_name=True)

    if dir_suffix:
        suffix = dir_suffix
    else:
        suffix = version

    # Train or evaluate or detect
    if command == "train":
        assert isinstance(train_epochs, int), "Epochs must be provided!"
        train(concrete_model, concrete_config, dataset, suffix, stage, train_epochs)
        # train(concrete_model, concrete_config, dataset, suffix, 2, train_epochs[1])
        # train(concrete_model, concrete_config, dataset, suffix, 3, train_epochs[2])
        # print("\n========training...")
    elif command == "val":
        if dir_suffix:
            print("Version will be ignored!")
            version = None
        # Validation dataset
        assert limit and validate_type, \
            "Limit and validate type must be provided!"
        val_dataset = ConcreteDataset()
        coco_data = val_dataset.load_concrete(dataset, val_subset, suffix, return_coco=True)
        val_dataset.prepare()
        if limit == 'all':
            limit = len(val_dataset.image_ids)
        print("Running concrete evaluation on {} images.".format(limit))
        evaluate_concrete(concrete_model, val_dataset, coco_data,
                          validate_type, limit=int(limit))
        print("\n=======val...")
    elif command == "detect":
        if detect_target == "subset":
            print("Detect images on test" + suffix + "....")
            multiple_detect(concrete_model, dataset, "test", suffix, save, csv)
            print("\n=======subset done!...")
        else:
            if concrete_config.NAME == "concrete":
                classes_name = ['BG', 'bughole']
            elif concrete_config.NAME == "coco":
                import concrete_mrcnn.demo as cd
                classes_name = cd.class_names
            single_detect(concrete_model, dataset, suffix, classes_name, val_subset,
                          image_name, random_image, verbose=1)
            print("\n=======images done!...")
    else:
        print("'{}' is not recognized. "
              "Use 'train', 'val' or 'detect'".format(command))

    # logs_clean(logs)


def precision_recall(config, mode, subset, version, weights, logs, gpu=None,
                     image_name=None, random_img=False, config_display=True,
                     verbose=True, image_display=True, threshold=0.5, curve=False,
                     compute_num=None, plot_overlaps=False, range_ap=False):
    assert mode in ["single", "batch"], "Compute on Single image or Batch images!"
    if config == "coco":
        inference_config = CocoConfig()
    elif config == "concrete":
        inference_config = InferenceConfig()
    if config_display:
        inference_config.display()

    if not gpu:
        device = "/cpu:0"
    else:
        assert gpu == 0 or 1, "Only two GPU available"
        device = "/gpu:{}".format(gpu)

    if inference_config.NAME == 'concrete':
        dataset = ConcreteDataset()
        dataset.load_concrete(DATASETS_PATH, subset, version)
    elif inference_config.NAME == "coco":
        dataset = coco.CocoDataset()
        dataset.load_coco(DATASETS_PATH, "val", "2017")
    dataset.prepare()
    if False:
        print("Images: {}\nClasses: {}".format(
            len(dataset.image_ids), dataset.class_names))

    with tf.device(device):
        model = modellib.MaskRCNN(mode="inference", model_dir=logs,
                                  config=inference_config)

    if inference_config.NAME == "concrete":
        assert weights, "A weights file path. Must be Provided!"
        if weights == "last":
            weights_path = model.find_last()
        else:
            weights_path = weights
    elif inference_config.NAME == "coco":
        weights_path = COCO_WEIGHTS_PATH

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    if mode == "single":
        assert random_img or image_name, "Random images or specific image!"
        if random_img:
            image_name = None
            image_id = random.choice(dataset.image_ids)
        else:
            for img_info in dataset.image_info:
                if img_info["path"].split('/')[-1] == image_name:
                    image_id == img_info["id"]

        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]

        # Run object detection
        results = model.detect([image], verbose=1)

        # Display results
        ax = get_ax(1)
        r = results[0]

        if image_display:
            visualization.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        dataset.class_names, r['scores'], ax=ax,
                                        title="Predictions")

        if verbose:
            print("\nimage ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                                   dataset.image_reference(image_id)))
            print("\nOriginal image shape: ",
                  modellib.parse_image_meta(image_meta[np.newaxis, ...])["original_image_shape"][0])

            log("\npred_class_ids", r['class_ids'])
            log("pred_rois", r['rois'])
            log("pred_masks", r['masks'])
            log("\ngt_class_id", gt_class_id)
            log("gt_bbox", gt_bbox)
            log("gt_mask", gt_mask)

        # Default average AP on IOU range 0.05-0.95
        ap, precisions, recalls, overlaps = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                                                         r['rois'], r['class_ids'], r['scores'], r['masks'],
                                                         verbose=1, iou_thresholds=threshold, curve=curve)
        visualize.plot_precision_recall(ap, precisions, recalls, threshold)

        if plot_overlaps:
            visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
                                    overlaps, dataset.class_names)

    else:
        assert isinstance(compute_num, int) and compute_num >= 0, \
            "How mang images do you wanna compute on?"
        if compute_num == 0:
            # compute on all images in given directory
            num =len(dataset.image_ids)
        else:
            if compute_num > len(dataset.image_ids):
                print("Compute number exceeds!Use the maximum number")
                num = len(dataset.image_ids)
            else:
                num == compute_num

        image_ids = np.random.choice(dataset.image_ids, num)
        aps, precisions, recalls, _ = compute_batch_ap(model, dataset, inference_config, image_ids,
                                                             threshold=threshold, verbose=0, curve=True)
        #Plot results
        visualize.plot_precision_recall(aps, precisions, recalls, threshold)

    # logs_clean(logs)

# Compute VOC-style Average Precision
def compute_batch_ap(model, dataset, config, image_ids, threshold,
                     verbose=0, curve=True):
    aps = []
    precision, recall = [], []
    wrong_count = 0
    print("Ready to evaluate on {} images ...".format(len(image_ids)))
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]

        try:
            ap, precisions, recalls, overlaps =\
                utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                                       r['rois'], r['class_ids'], r['scores'], r['masks'],
                                       iou_thresholds=threshold, verbose=0, curve=curve)
        except:
            wrong_count += 1
            print("No.{} Wrong image:{}({})".format(wrong_count,
                                                dataset.image_info[image_id]["id"],
                                                image_id))
            continue
        aps.append(ap)
        if curve:
            precision.append(precisions)
            recall.append(recalls)

        if verbose:
            if isinstance(threshold, float):
                print("{} ==> mAP @IoU={:.2f}: ".format(
                    dataset.image_info[image_id]['path'].split('/')[-1], threshold), ap)
            else:
                print("{} ==> mAP @IoU={:.2f}~{:.2f}: ".format(
                    dataset.image_info[image_id]['path'].split('/')[-1],
                    float(threshold[0]), float(threshold[-1]), ap))
    if wrong_count:
        print("Total number of wrong images:{}\n".format(wrong_count))

    aps = np.array(aps).mean()
    if curve:
        recall, precision = format_interpolate(recall, precision)

    if not isinstance(threshold, float):
        print("mAP @IoU={:.2f}-{:.2f}:\t {:.3f}".format(float(threshold[0]),
                                                        float(threshold[-1]), np.mean(aps)))
    else:
        print("mAP @IoU={:.2f}: ".format(threshold), np.mean(aps))

    return aps, precision, recall, overlaps


def format_interpolate(r, p):
    assert isinstance(r, list), "Recalls must be list type!"
    r = np.array(r)
    p = np.array(p)
    num = max([r[i].shape[0] for i in range(r.shape[0])])
    r_new = np.linspace(0., 1., num)
    p_news = np.zeros([len(r), num])
    for i in range(len(r)):
        f = interpolate.interp1d(r[i], p[i], kind='slinear')
        p_new = f(r_new)
        p_news[i, :] = p_new
    assert p_news.shape == (len(r), num)
    p_new = np.mean(p_news, axis=0)

    return r_new, p_new

def display_differences(config, dirname, config_display=True, device = "/gpu:0"):
    if config == "coco":
        inference_config = CocoConfig()
        dataset = coco.CocoDataset()
        dataset.load_coco(DATASETS_PATH, "val", "2017")
    elif config == "concrete":
        inference_config = InferenceConfig()
        dataset = ConcreteDataset()
        dataset.load_concrete(DATASETS_PATH, force_path=dirname)
    if config_display:
        inference_config.display()
    dataset.prepare()

    with tf.device(device):
        model = modellib.MaskRCNN(mode="inference",
                                  model_dir=DEFAULT_LOGS_DIR,
                                  config=inference_config)

    if config == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif config == "concrete":
        weights_path = model.find_last()
    print("\nLoading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    image_id = random.choice(dataset.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("\nimage ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset.image_reference(image_id)))
    print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis, ...])["original_image_shape"][0])

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    r = results[0]
    log("\ngt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # Compute AP over range 0.5 to 0.95 and print it
    utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                           r['rois'], r['class_ids'], r['scores'], r['masks'],
                           verbose=1, curve=True, iou_thresholds=None)

    visualize.display_differences(
        image,
        gt_bbox, gt_class_id, gt_mask,
        r['rois'], r['class_ids'], r['scores'], r['masks'],
        dataset.class_names, ax=get_ax(),
        show_box=False, show_mask=False,
        iou_threshold=0.5, score_threshold=0.5)

    # Display predictions only
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             dataset.class_names, r['scores'], ax=get_ax(1),
    #                             show_bbox=False, show_mask=False,
    #                             title="Predictions")

    # Display Ground Truth only
    # visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
    #                             dataset.class_names, ax=get_ax(1),
    #                             show_bbox=False, show_mask=False,
    #                             title="Ground Truth")

    # logs_clean(logs)


def logs_clean(path):
    files = os.listdir(path)
    # root = os.path.abspath(path)
    # print('\n...To clean logs in', path)
    for f in files:
        # print(f)
        f_path = os.path.join(path, f)
        # print(f_path)
        if os.path.isdir(f_path):
            if not os.listdir(f_path):
                # print("del dir")
                os.rmdir(f_path)
                # print('==> Empty logs dir clean done in {}\n'.format(path))
        elif os.path.isfile(f_path):
            if os.path.getsize(f_path) == 0:
                # print("del file")
                os.remove(f_path)
                # print('==> Empty files clean done in {}\n'.format(path))


def concrete_train_schedule(epochs, version, stage=[1, 1, 1]):
    # logs_clean(DEFAULT_LOGS_DIR)
    if stage[0]:
        print("\n=== Stage 1 ===>")
        # Stage 1
        train_strategy(command="train", weights="coco", config_display=True,
                       dataset=DATASETS_PATH, version=version, logs=DEFAULT_LOGS_DIR,
                       stage="1", train_epochs=epochs[0],    # train
                       dir_suffix=None,   # specific target directory
                       limit=None, validate_type=None,    # val
                       save=False, csv=False, val_subset=None,   # detect multiple images and save
                       detect_target=None, classes_name=None, # detect single image
                       image_name=None, random_image=False)    # or random image
    if stage[1]:
        print("\n=== Stage 2 ===>")
        # Stage 2
        train_strategy(command="train", weights="last", config_display=False,
                       dataset=DATASETS_PATH, version=version, logs=DEFAULT_LOGS_DIR,
                       stage="2", train_epochs=epochs[1],  # train
                       dir_suffix=None,  # specific target directory
                       limit=None, validate_type=None,  # val
                       save=False, csv=False, val_subset=None,  # detect multiple images and save
                       detect_target=None, classes_name=None,  # detect single image
                       image_name=None, random_image=False)  # or random image
    if stage[2]:
        print("\n=== Stage 3 ===>")
        # Stage 3
        train_strategy(command="train", weights="last", config_display=False,
                       dataset=DATASETS_PATH, version=version, logs=DEFAULT_LOGS_DIR,
                       stage="3", train_epochs=epochs[2],  # train
                       dir_suffix=None,  # specific target directory
                       limit=None, validate_type=None,  # val
                       save=False, csv=False, val_subset=None,  # detect multiple images and save
                       detect_target=None, classes_name=None,  # detect single image
                       image_name=None, random_image=False)  # or random image


class Trainer:
    def __init__(self, model, dataset_train, dataset_val, config, augmentation):
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.config = config
        self.augmentation = augmentation
        self.total_epochs = 0

    def train(self, message, num_epochs, layers, learning_rate_factor=1.0):
        self.total_epochs += num_epochs
        print(message + f' for {num_epochs} mini-epochs.')
        self.model.train(self.dataset_train, self.dataset_val,
                         learning_rate=self.config.LEARNING_RATE * learning_rate_factor,
                         epochs=self.total_epochs,
                         layers=layers,
                         augmentation=self.augmentation)


def training_steps():
    trainer = Trainer(model, dataset_train, dataset_val, config, augmentation)

    trainer.train('Training Stage 1: only network heads',
                  num_epochs=1, layers='heads')

    trainer.train('Training Stage 2: warmup for network except resnet-50',
                  num_epochs=1, layers='4+', learning_rate_factor=0.01)

    trainer.train('Training Stage 3: train network except resnet-50',
                  num_epochs=5, layers='4+')

    trainer.train('Training Stage 4: warmup for everything',
                  num_epochs=1, layers='all', learning_rate_factor=0.01)

    trainer.train('Training Stage 5: train everything',
                  num_epochs=100, layers='all')

    trainer.train('Training Stage 6: fine-tune / 10',
                  num_epochs=20, layers='all', learning_rate_factor=0.1)

    trainer.train('Training Stage 7: fine-tune / 100',
                  num_epochs=10, layers='all', learning_rate_factor=0.01)






