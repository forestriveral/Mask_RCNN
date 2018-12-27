
import os
import sys
import cv2
import random
import copy
import json
import colorsys
import collections
import pandas as pd
import numpy as np
from scipy import interpolate
from interval import Interval
from sklearn.metrics import auc
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines, rcParams
from matplotlib.patches import Polygon
from mrcnn import model as modellib, utils
from mrcnn import visualize
# from concrete_mrcnn.concrete import get_ax
# import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None, save=False,
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
                color='w', size=10, backgroundcolor="None")

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

    if save:
        plt.margins(0, 0)
        plt.savefig('./detected_image.png', dpi=300, bbox_inches='tight')
        print("Save done!")

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
                  limit=None, types="bbox", save=False):
    # Pick COCO images from the dataset
    if image_ids:
        image_ids = image_ids
        limit = None
    else:
        image_ids = dataset.image_ids

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
    print("Ready to formatting on {} ...".format(types))
    for i, image_id in enumerate(image_ids):
        # Load annotations gt information
        gt, gt_num, image = build_voc_gts(dataset, config, image_id, class_id,
                                          types, gt, gt_num)
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


def voc_ap_compute(dataset, class_id, detection, groundtruth, gt_num,
                   class_names=None, types="bbox", threshold=0.5,
                   load=False, debug=False):
    if load:
        gt_name = "./{}_gt.json".format(types)
        with open(gt_name, 'r', encoding='utf-8') as f:
            print("Loading gt annotation file ...")
            groundtruth = json.load(f)

        result_name = "./{}_results.json".format(types)
        with open(result_name, 'r', encoding='utf-8') as f:
            print("Loading detection results file ...")
            detection = json.load(f)

    print("Ready to evaluate on {} ...".format(types))
    # Copy gt file
    gt = copy.deepcopy(groundtruth)
    # Loop for every class need to compute ap
    evaluation = ["precisions", "recalls", "maps", "fprs", "tprs", "aucs"]
    targets = {}
    for eva in evaluation:
        targets[eva] = {}
    # precisions, recalls, maps, fprs, tprs, aucs = {}, {}, {}, {}, {}, {}
    for i, cls in enumerate(class_id):
        # If there is no instance of this class detected
        if (not detection[i]["region"]) or (not detection[i]["confidence"]):
            # detection[i] = {}
            continue
        else:
            # While class_name = None means all classes
            if class_names is None:
                class_names = dataset.class_names[1:]
            for k in targets.keys():
                targets[k][class_names[i]] = []

            # sort by confidence
            sorted_ind = np.argsort(-1 * np.array(detection[i]["confidence"]))
            region = np.array(detection[i]["region"])[sorted_ind, :] if types == "bbox" else \
                np.array(detection[i]["region"]).transpose((1, 2, 0))[..., sorted_ind]
            image_ids = [detection[i]["image_ids"][x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros((len(threshold), nd)) if isinstance(threshold, list) and len(threshold) > 1 else \
                np.zeros(nd)
            fp = np.zeros((len(threshold), nd)) if isinstance(threshold, list) and len(threshold) > 1 else \
                np.zeros(nd)
            for d in range(nd):
                if image_ids[d] not in gt[i].keys():
                    if isinstance(threshold, (list, np.ndarray)):
                        assert len(threshold) > 1
                        fp[:, d] = 1
                    else:
                        fp[d] = 1
                    # print('No gt instances but detected out:', image_ids[d])
                    continue
                r = gt[i][image_ids[d]]
                bb = region[d, :].astype(np.float32)[None, ...] if types == "bbox" \
                    else region[:, :, d][:, :, None]
                ovmax = -np.inf
                bbgt = np.array(r['region']).astype(np.float32) if types == "bbox" \
                    else np.array(r['region']).transpose((1, 2, 0))

                if bbgt.size > 0:
                    # compute overlaps
                    # overlaps = voc_overlaps(bbgt, gt)
                    overlaps = utils.compute_overlaps(bbgt, bb).transpose(1, 0) if types == "bbox" else \
                        utils.compute_overlaps_masks(bbgt, bb).transpose(1, 0)
                    assert overlaps.shape == (1, bbgt.shape[0]) if types == "bbox" \
                        else (1, bbgt.shape[-1])
                    ovmax = np.max(np.squeeze(overlaps))
                    jmax = np.argmax(np.squeeze(overlaps))

                if debug:
                    if d == debug:
                        debug_tools(i, r)
                        debug_tools(i, bb.shape)
                        debug_tools(i, bbgt.shape)

                        debug_tools(i, overlaps)
                        debug_tools(i, ovmax)
                        debug_tools(i, jmax)

                if isinstance(threshold, (list, np.ndarray)) and len(threshold) > 1:
                    if len(r['det']) == 1 or not isinstance(r['det'][0], list):
                        r['det'] = [copy.deepcopy(r['det']) for _ in range(len(threshold))]
                        assert r['det'][1] == [False] * len(r['det'][1])
                    for ind, t in enumerate(threshold):
                        if ovmax > t:
                            if not r['det'][ind][jmax]:
                                tp[ind, d] = 1.
                                r['det'][ind][jmax] = 1
                            else:
                                fp[ind, d] = 1.
                        else:
                            fp[ind, d] = 1.
                else:
                    assert isinstance(threshold, (float, list))
                    if isinstance(threshold, list) and len(threshold) == 1:
                        threshold = threshold[0]
                    if ovmax > threshold:
                        if not r['det'][jmax]:
                            tp[d] = 1.
                            r['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                    else:
                        fp[d] = 1.

                if debug:
                    if d == debug:
                        debug_tools(i, np.count_nonzero(tp))
                        debug_tools(i, np.count_nonzero(fp))
                        debug_tools(i, r['det'])

            # Check results
            # assert int(fp[-1] + tp[-1]) == region.shape[0]
            # compute precision recall
            # print("fp", fp)
            # print("tp", tp)
            fp = np.cumsum(fp) if fp.ndim == 1 else np.cumsum(fp, axis=1)
            tp = np.cumsum(tp) if tp.ndim == 1 else np.cumsum(tp, axis=1)
            # compute true positive rate and false negative
            fpr = fp / fp[-1] if fp.ndim == 1 else fp / fp[:, -1].reshape(fp.shape[0], 1)
            tpr = tp / tp[-1] if tp.ndim == 1 else tp / tp[:, -1].reshape(tp.shape[0], 1)

            # Compute area under curve
            area = np.array([auc(fpr, tpr)]) if tp.ndim == 1 else \
                np.array([auc(fpr[ix, :], tpr[ix, :]) for ix in range(tp.shape[0])])
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            recall = tp / float(gt_num[i])
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            if recall.ndim == 1:
                ap, prec, rec = voc_ap(recall, precision, use_07_metric=False)
                ap, prec, rec = np.array([ap]), prec[None, :], rec[None, :]
            else:
                ap = np.zeros([fp.shape[0]])
                prec = np.zeros([precision.shape[0], precision.shape[1] + 2])
                rec = np.zeros([recall.shape[0], recall.shape[1] + 2])
                for ix in range(recall.shape[0]):
                    a, p, r = voc_ap(recall[ix, :], precision[ix, :], use_07_metric=False)
                    ap[ix] = a
                    prec[ix, :] = p
                    rec[ix, :] = r

            print("\nFP:", fp.shape)
            print("TP:", tp.shape)
            print("FPR:", fpr.shape)
            print("TPR:", tpr.shape)
            print("AREA:", area, area.shape)
            print("AP", ap, ap.shape)
            if debug:
                # Debug
                # print("GT: ", gt[0])
                print("\nFP: ", fp)
                print("TP: ", tp)
                print("Recall: ", len(recall))
                print("Precision: ", len(precision))
                print("AP: ", ap)
                print("Area under curve: ", area)
                # print("FPR: ", fpr)
                # print("TPR: ", tpr)

            # Pad with start and end values to simplify the math
            fpr = np.pad(fpr, ((0, 0), (1, 1)), "constant", constant_values=(0, 1)) if fpr.ndim > 1 \
                else np.pad(fpr, (1, 1), "constant", constant_values=(0, 1))[None, :]
            tpr = np.pad(tpr, ((0, 0), (1, 1)), "constant", constant_values=(0, 1)) if tpr.ndim > 1 \
                else np.pad(tpr, (1, 1), "constant", constant_values=(0, 1))[None, :]

        for (x, y) in zip(evaluation, [prec, rec, ap, fpr, tpr, area]):
            targets[x][class_names[i]] = y

    # Clean the class that has no instances detected
    # detection = clean_duplicates(detection)
    # Zipped dict storing recalls, precisions, maps, fprs, tprs, aucs
    return targets


def debug_tools(i, output):
    if i == 0:
        print("\nOutput:\n{}".format(output))
        if isinstance(output, np.ndarray):
            print("Shape:\n{}".format(output.shape))
        if isinstance(output, (list, dict)):
            print("Length:\n{}".format(len(output)))
        # print("Type:\n{}".format(type(output)))


def voc_overlaps(gt, det):
    # compute overlaps
    # intersection
    ixmin = np.maximum(gt[:, 0], det[0])
    iymin = np.maximum(gt[:, 1], det[1])
    ixmax = np.minimum(gt[:, 2], det[2])
    iymax = np.minimum(gt[:, 3], det[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((det[2] - det[0] + 1.) * (det[3] - det[1] + 1.) +
           (gt[:, 2] - gt[:, 0] + 1.) *
           (gt[:, 3] - gt[:, 1] + 1.) - inters)
    overlaps = inters / uni
    return overlaps


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
            if image_id not in gt[ind].keys():
                gt[ind][image_id] = {'region': [], 'det': []}
            gt[ind][image_id]['region'].append(gt_bbox[i, :] if types == "bbox" else gt_mask[:, :, i])
            gt[ind][image_id]['det'].append(False)
            count[ind] += 1
        else:
            continue
    return gt, count, image


def build_voc_results(r, results, class_id, image_id, types, count):
    if r["rois"] is None:
        pass
    assert r["rois"].shape[0] == r["class_ids"].shape[0]
    assert r["scores"].shape[0] == r["masks"].shape[-1]

    for i, idx in enumerate(r["class_ids"]):
        if idx in class_id:
            ind = list(class_id).index(idx)
            results[ind]["image_ids"].append(image_id)
            results[ind]["confidence"].append(float(r["scores"][i]))
            results[ind]["region"].append(r["rois"][i, :] if types == "bbox" else r["masks"][:, :, i])
            count[ind] += 1
        else:
            continue
    return results, count


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


def print_data_info(*args):
    for v in args:
        print(v.shape, type(v))


def plot_voc_curve(data, classes, threshold, curve=None,
                   fs=8, save=False, name="curve"):
    if curve and isinstance(curve, str):
        curve = [curve]
    else:
        curve = curve or ["pr", "roc"]
    if isinstance(classes, str):
        classes = [classes]
    for c in classes:
        pre, rec, aps, fpr, tpr, aucs = parse_evaluation(data, c)
        print_data_info(pre, rec, aps, fpr, tpr, aucs)
        if "pr" in curve:
            plot_evaluate_curve(aps, pre, rec, threshold=threshold, curve="pr",
                                fs=fs, save=save, filename=name + "_pr")
        if "roc" in curve:
            plot_evaluate_curve(aucs, tpr, fpr, threshold=threshold, curve="roc",
                                fs=fs, save=save, filename=name + "_roc")


def parse_evaluation(cache, cla):
    precision = cache["precisions"]
    recall = cache["recalls"]
    maps = cache["maps"]
    fpr = cache["fprs"]
    tpr = cache["tprs"]
    aucs = cache["aucs"]

    aps, pre, rec = maps[cla], precision[cla], recall[cla]
    aucs, fpr, tpr = aucs[cla], fpr[cla], tpr[cla]

    return pre, rec, aps, fpr, tpr, aucs


def plot_evaluate_curve(ap, p, r, threshold=None, curve="pr",
                        fs=8, save=False, filename="curve"):
    linestyles = ['-', '--', ':', '-.', '-']
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 20,
            }
    font_legend = {'family': 'Times New Roman',
                   'weight': 'bold',
                   'size': 15,
                   }
    colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
               (0.0, 1.0, 0.0), (1.0, 0.0, 1.0),
               (1.0, 1.0, 0.0)]

    threshold = threshold or 0.5
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1, figsize=(fs, fs))

    # Plot P-R curve
    if curve == "pr":
        if p.shape[0] == 1:
            print("\nSingle {} line".format(curve))
            if isinstance(threshold, list):
                threshold = threshold[0]
            ax.set_title("Precision-Recall Curve. AP@IoU {:.2f} = {:.3f}".format(threshold, ap[0]), font)
            _ = ax.plot(r[0], p[0], ls='-', c='#CD0000', lw=1.5)
        else:
            print("\nMultiple {} lines".format(curve))
            assert isinstance(threshold, list), \
                "Multiple thresholds should be provided!"
            ax.set_title("Precision-Recall Curve", font)
            for i in range(len(p)):
                _ = ax.plot(r[i], p[i], ls=linestyles[i],
                            c=colors[i], lw=1.5,
                            label="AP={:.3f}(IoU={:.2f})".format(ap[i], threshold[i]))
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1.05)
        plt.xlabel("Recall", font)
        plt.ylabel("Precision", font)
        plt.legend(loc="best", prop=font_legend,
                   edgecolor='None', frameon=False,
                   labelspacing=0.4)
    # Plot ROC curve
    if curve == "roc":
        if p.shape[0] == 1:
            print("\nSingle {} line".format(curve))
            if isinstance(threshold, list):
                threshold = threshold[0]
            _ = ax.plot(r[0], p[0], ls='-', c='#CD0000', lw=1.5,
                        label="AUC={:.3f}(IoU={:.2f})".format(ap[0], threshold))
        else:
            print("\nMultiple {} lines".format(curve))
            assert isinstance(threshold, list), \
                "Multiple thresholds should be provided!"
            for i in range(p.shape[0]):
                _ = ax.plot(r[i], p[i], ls="-",
                            c=colors[i], lw=1.5,
                            label="AUC={:.3f}(IoU={:.2f})".format(ap[i], threshold[i]))

        # Standard line
        ax.plot([0, 1], [0, 1], ls='--', c='#778899', lw=1.5, label="Standard line")
        # Set plot title
        ax.set_title("ROC Curve", font)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 1.0)
        plt.xlabel("False Positive Rate (FPR)", font)
        plt.ylabel("True Positive Rate (TPR)", font)
        plt.legend(loc="lower right", prop=font_legend,
                   edgecolor='None', frameon=False,
                   labelspacing=0.4)

    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # ax.set_xticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    ax.set_xticklabels(('0', '0.2', '0.4', '0.6', '0.8', '1.0'))
    ax.set_yticklabels(('', '0.2', '0.4', '0.6', '0.8', '1.0'))
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'

    if save:
        plt.margins(0, 0)
        plt.savefig('./{}.png'.format(filename), dpi=300, bbox_inches='tight')
        print("Save done!")

    plt.show()


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
