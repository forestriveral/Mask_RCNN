
import os
import sys
import cv2
import random
import json
import colorsys
import collections
import pandas as pd
import numpy as np
from interval import Interval
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from mrcnn import model as modellib, utils
# from concrete_mrcnn.concrete_tools import build_concrete_results
# import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True, show_group=False,
                      colors=None, captions=None, threshold=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors or colors group
    if not show_group:
        colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    area_sum = 0
    rate_sum = 0
    for i in range(N):
        # Mask
        mask = masks[:, :, i]
        ignore, area, rate, diameter = mask_area_compute(mask,
                                                         threshold=threshold)
        area_sum += area
        rate_sum += rate
        if ignore:
            continue

        if show_group:
            color = assign_color_group(rate, colors=None)
        else:
            color = colors[i]

        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            # x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}|{:.2f}".format(label, score, rate) \
                if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 - 5, caption,
                color='b', size=10, backgroundcolor="None")

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()

    return area_sum, rate_sum


def exclude_ignore_instance(boxes, masks, class_ids,
                            scores=None, threshold=None):
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to exclude *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    keep_ix = np.zeros([10], dtype=bool)
    for i in range(N):
        # Mask
        mask = masks[:, :, i]
        ignore, area, rate, diameter = mask_area_compute(mask,
                                                         threshold=threshold)
        if ignore:
            continue
        else:
            keep_ix[i] = 1
    boxes = boxes[keep_ix]
    masks = masks[:, :, keep_ix]
    class_ids = class_ids[keep_ix]
    scores = scores[keep_ix]

    return boxes, masks, class_ids, scores, keep_ix


def mask_area_compute(mask, threshold=None):
    threshold = threshold or 0.0
    mask = mask.astype(np.uint8) * 255
    image_area = float(mask.shape[:2][0] * mask.shape[:2][1])
    _, contours, hireachy = cv2.findContours(mask,
                                             cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
    assert isinstance(contours, list), "Invalid contours format!"
    if len(contours) > 1:
        total_area = 0
        total_rate = 0
        diameters = []
        for contour in contours:
            mask_area, area_rate, diameter = contour_compute(contour, image_area)
            total_area += mask_area
            diameters.append(diameter)
            total_rate += area_rate
        if total_rate > threshold:
            ignore = False
        else:
            ignore = True
        return ignore, total_area, total_rate, np.max(np.array(diameters))
    elif len(contours) == 1:
        contour = contours[0]
        mask_area, area_rate, diameter = contour_compute(contour, image_area)
        if area_rate > threshold:
            ignore = False
        else:
            ignore = True
        return ignore, mask_area, area_rate, diameter
    else:
        raise ValueError("No contours in mask!")


def contour_compute(contour, image_area, scale=None):
    scale = scale or 1000 * 100
    mask_area = abs(cv2.contourArea(contour))
    max_diameter = np.sqrt(4 * mask_area / np.pi)
    area_rate = mask_area / image_area * scale
    return mask_area, area_rate, max_diameter


def voc_ap_format(model, config, dataset, image_ids=None, class_names=None,
                  limit=None, types="mask", save=False):
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = random.sample(list(image_ids), limit)

    assert isinstance(image_ids, (list, int)),\
        "Images list or selected image!"
    if isinstance(image_ids, int):
        print("Evaluate on Image ID {}".format(image_ids))
        image_ids = [image_ids]

    cls = dataset.class_names
    if class_names is not None:
        # class_id = [c["id"] for c in dataset.class_info if c["name"] in class_name]
        class_id = [cls.index(c) for c in class_names]
        assert len(class_id) == len(class_names), "No repeat class name!"
    else:
        class_id = dataset.class_ids[1:]

    # Initiate the format to store gt or detection results
    results = [{"image_ids": [],
                "confidence": [],
                "region": []} for _ in range(len(class_id))]
    results_num = [0 for _ in range(len(class_id))]

    gt = [{} for _ in range(len(class_id))]
    gt_num = [0 for _ in range(len(class_id))]
    for i, image_id in enumerate(image_ids):
        # Load annotations gt information
        gt, gt_num = build_voc_gts(dataset, config, image_id, class_id,
                                   types, gt, gt_num)
        # Load image
        image = dataset.load_image(image_id)
        # Run detection
        r = model.detect([image], verbose=0)[0]
        # Build the voc results
        results, results_num = build_voc_results(r, results, class_id,
                                                 image_id, types, results_num)
    # Check whether instances of each class exist or not?
    assert len(results) == len(gt)
    no_instance_class = [[], []]
    for i, (x, y) in enumerate(zip(results, gt)):
        if not x["region"]:
            no_instance_class[0].append(cls[class_id[i]])
        if not y:
            no_instance_class[1].append(cls[class_id[i]])
    if no_instance_class[0]:
        print("No instances of following class are detected:\n",
              no_instance_class[0] if len(no_instance_class[0]) < 10 else len(no_instance_class[0]))
        if len(no_instance_class[0]) < 10:
            print("Detected classes: \n{}".format(set(cls[1:]) - set(no_instance_class[0])))
    if no_instance_class[1]:
        print("No groundtruth of following class are found\n",
              no_instance_class[1] if len(no_instance_class[1]) < 10 else len(no_instance_class[1]))
        if len(no_instance_class[1]) < 10:
            print("Groundtruth classes: \n{}".format(set(cls[1:]) - set(no_instance_class[1])))

    if save:
        gt_name = "./{}_gt.json".format(types)
        with open(gt_name, 'w', encoding='utf-8') as f:
            json.dump(gt, f)
            print("\nGt annotation file saved done! "
                  "Gt instances number: {}\n".format(gt_num))

        result_name = "./{}_results.json".format(types)
        with open(result_name, 'w', encoding='utf-8') as f:
            json.dump(results, f)
            print("\nDetection results file saved done! "
                  "Detected instances number: {}\n".format(results_num))

    return class_id, results, gt, gt_num, results_num


def voc_ap_compute(dataset, class_id, results, gt, gt_num,
                   class_names=None, types="bbox", threshold=0.5,
                   load=False):
    if load:
        gt_name = "./{}_gt.json".format(types)
        with open(gt_name, 'r', encoding='utf-8') as f:
            print("Loading gt annotation file ...")
            gt = json.load(f)

        result_name = "./{}_results.json".format(types)
        with open(result_name, 'r', encoding='utf-8') as f:
            print("Loading detection results file ...")
            results = json.load(f)

    # Check files

    # Loop for every class need to compute ap
    precisions = {}
    recalls = {}
    maps = {}
    for i, cls in enumerate(class_id):
        # If there is no instance of this class detected
        if (not results[i]["region"]) or (not results[i]["confidence"]):
            # results[i] = {}
            continue
        else:
            # While class_name = None means all classes
            if class_names is None:
                class_names = dataset.class_names[1:]
            precisions[class_names[i]] = []
            recalls[class_names[i]] = []
            maps[class_names[i]] = []

            # sort by confidence
            sorted_ind = np.argsort(-1 * np.array(results[i]["confidence"]))
            region = np.array(results[i]["region"])[sorted_ind, :] if types == "bbox" else \
                np.array(results[i]["region"])[:, :, sorted_ind]
            image_ids = [results[i]["image_ids"][x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                if not image_ids[d] in gt[i]:
                    fp[d] = 1
                    # print('No gt instances but detected out:', image_ids[d])
                    continue

                R = gt[i][image_ids[d]]
                bb = region[d, :].astype(float)[None, ...] if types == "bbox" \
                    else region[:, :, d][..., None]
                ovmax = -np.inf
                BBGT = np.array(R['region']).astype(float)

                if BBGT.size > 0:
                    # compute overlaps
                    overlaps = utils.compute_overlaps(BBGT, bb).transpose(1, 0) if types == "bbox" else \
                        utils.compute_overlaps_masks(BBGT, bb).transpose(1, 0)
                    assert overlaps.shape == (1, BBGT.shape[0])
                    ovmax = np.max(np.squeeze(overlaps))
                    jmax = np.argmax(np.squeeze(overlaps))

                if isinstance(threshold, (list, np.ndarray)):
                    pass
                else:
                    assert isinstance(threshold, (float, int))
                    if ovmax > threshold:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                    else:
                        fp[d] = 1.

                # compute precision recall
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                recall = tp / float(gt_num[i])
                # avoid divide by zero in case the first detection matches a difficult
                # ground truth
                precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                ap, prec, rec = voc_ap(recall, precision, use_07_metric=False)

            precisions[class_names[i]].append(list(prec))
            recalls[class_names[i]].append(list(rec))
            maps[class_names[i]].append(float(ap))

    # Clean the class that has no instances detected
    # results = clean_duplicates(results)

    return recalls, precisions, maps


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        mpre, mrec = [], []
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, mpre, mrec


def build_voc_gts(dataset, config, image_id, class_id,
                  types, gt, count):
    # Load annotations gt information
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config, image_id)
    for i, idx in enumerate(gt_class_id):
        if idx in class_id:
            ind = list(class_id).index(idx)
            # if types == "bbox":
            #     target = gt_bbox[i]
            # if types == "mask":
            #     target = gt_mask[:, :, i]
            if image_id not in gt[ind].keys():
                gt[ind][image_id] = {'region': [], 'det': []}
            gt[ind][image_id]['region'].append(gt_bbox[i, :] if types == "bbox" else gt_mask[:, :, i])
            gt[ind][image_id]['det'].append(False)
            count[ind] += 1
        else:
            continue
    return gt, count


def build_voc_results(r, results, class_id, image_id, types, count):
    if r["rois"] is None:
        pass
    assert r["rois"].shape[0] == r["class_ids"].shape[0] \
           == r["scores"].shape[0] == r["masks"].shape[-1]

    for i, idx in enumerate(r["class_ids"]):
        if idx in class_id:
            ind = list(class_id).index(idx)
            results[ind]["image_ids"].append(image_id)
            results[ind]["confidence"].append(float(r["scores"][i]))
            # if types == "bbox":
            #     target = r["rois"][i]
            # if types == "mask":
            #     target = r["masks"][:, :, i]
            results[ind]["region"].append(r["rois"][i, :] if types == "bbox" else r["masks"][:, :, i])
            count[ind] += 1
        else:
            continue
    return results, count


def clean_duplicates(seq, remove):
    l1 = seq
    c = collections.Counter(l1)
    l2 = sorted(set(l1), key=l1.index)
    l2.remove(remove)
    for k, v in c.items():
        if k != remove and v != 1:
            raise ValueError("Invalid list!")
    if len(l1) - len(l2) != c[-1]:
        raise ValueError("Invalid list!")
    else:
        return l2


def assign_color_group(value, intervals=None, colors=None):
    intervals = intervals or [10, 33, 56, 79, 100]
    group = [(0.0, 1.0, 0.0), (0.0, 0.0, 0.95), (1.0, 1.0, 0.0),
              (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)]
    colors = colors or group

    if value < intervals[0]:
        color = (1.0, 1.0, 1.0)
        return color
    elif value in Interval(intervals[0], intervals[1]):
        return colors[0]
    elif value in Interval(intervals[1], intervals[2]):
        return colors[1]
    elif value in Interval(intervals[2], intervals[3]):
        return colors[2]
    elif value in Interval(intervals[3], intervals[4]):
        return colors[3]
    elif value > intervals[4]:
        return colors[4]


def group_colors():
    pass


def random_colors(N, bright=True, shuffle=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if shuffle:
        random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def plot_loss_curve(path, figsize=(16, 16), save=False, save_path=None):
    data = pd.read_csv(path)
    data = data.to_dict(orient='list')

    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 20,
            }
    font_legend = {'family': 'Times New Roman',
                   'weight': 'bold',
                   'size': 15,
                   }

    plt.figure(figsize=figsize)
    plt.subplots()
    plt.plot(data['epoch'], data['loss'], ls='-',
             c='#CD0000', lw=1.5, label="Training")
    plt.plot(data['epoch'], data['val_loss'], ls='--',
             c='#66CD00', lw=1.5, label="Validation")
    plt.xlim(0, len(data['epoch']))
    plt.xlabel('Epochs', font)
    plt.ylabel('Loss', font)

    ax = plt.gca()
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # ax.spines['top'].set_color('none')
    # ax.spines['right'].set_color('none')
    # ax.yaxis.grid(True, which='major')

    plt.legend(loc="upper right", prop=font_legend,
               edgecolor='None', frameon=False,
               labelspacing=0.2)
    if save:
        assert save_path, "Path to save must be provided!"
        plt.margins(0, 0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multi_loss_curve(data, figsize=(16, 16),
                          save=False, save_path=None):
    linestyles = ['-', '--', ':', '-.', '-']
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 20,
            }
    font_legend = {'family': 'Times New Roman',
                   'weight': 'bold',
                   'size': 15,
                   }
    colors = [(0.0, 1.0, 0.0), (0.0, 0.0, 0.95),
              (1.0, 1.0, 0.0), (1.0, 0.0, 0.0),
              (1.0, 0.0, 1.0)]

    plt.figure(figsize=figsize)
    plt.subplots()
    for i in range(5):
        loss = list(data.columns.values)[i+2]
        label = ' '.join(list(data.columns.values)[i+2].split('_'))
        plt.plot(data['epoch'], data[loss], ls=linestyles.pop(), c=colors.pop(),
                 lw=1.5, label=label)
    plt.xlim(0, len(data['epoch'])), plt.xlabel('Epochs', font), plt.ylabel('Loss', font)

    ax = plt.gca()
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # ax.spines['top'].set_color('none')
    # ax.spines['right'].set_color('none')
    # ax.yaxis.grid(True, which='major')

    plt.legend(loc="best", prop=font_legend,
               edgecolor='None', frameon=False,
               labelspacing=0.2)
    if save:
        assert save_path, "Path to save must be provided!"
        plt.margins(0, 0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
