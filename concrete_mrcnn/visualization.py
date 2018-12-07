
import os
import sys
import cv2
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils


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
        ignore, area, rate, diameter = mask_area_compute(i, mask,
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
            x = random.randint(x1, (x1 + x2) // 2)
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
        ignore, area, rate, diameter = mask_area_compute(i, mask,
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


def mask_area_compute(i, mask, threshold=None):
    threshold =threshold or 0.0
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
        raise "No contours in mask!"


def contour_compute(contour, image_area, scale=None):
    scale = scale or 1000 * 100
    mask_area = abs(cv2.contourArea(contour))
    max_diameter = np.sqrt(4 * mask_area / np.pi)
    area_rate = mask_area / image_area * scale
    return mask_area, area_rate, max_diameter


def assign_color_group(value, intervals=None, colors=None):
    intervals = intervals or [10, 33, 56, 79, 100]
    group = [(0.0, 1.0, 0.0), (0.0, 0.0, 0.95), (1.0, 1.0, 0.0),
              (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)]
    colors = colors or group

    if value < intervals[0]:
        color = (1.0, 1.0, 1.0)
    elif value > intervals[0] and value < intervals[1]:
        color = colors[0]
    elif value > intervals[1] and value < intervals[2]:
        color = colors[1]
    elif value > intervals[2] and value < intervals[3]:
        color = colors[2]
    elif value > intervals[3] and value < intervals[4]:
        color = colors[3]
    elif value > intervals[4]:
        color = colors[4]
    return color


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
            'size': 20,}
    font_legend = {'family': 'Times New Roman',
                   'weight': 'bold',
                   'size': 15,}

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
    #ax.spines['top'].set_color('none')
    #ax.spines['right'].set_color('none')
    #ax.yaxis.grid(True, which='major')

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
        loss = list(df.columns.values)[i+2]
        label = ' '.join(list(df.columns.values)[i+2].split('_'))
        plt.plot(data['epoch'], data[loss], ls=linestyles.pop(), c=colors.pop(),
                 lw=1.5, label=label)
    plt.xlim(0, len(data['epoch'])), plt.xlabel('Epochs', font), plt.ylabel('Loss', font)

    ax = plt.gca()
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    #ax.spines['top'].set_color('none')
    #ax.spines['right'].set_color('none')
    #ax.yaxis.grid(True, which='major')

    plt.legend(loc="best", prop=font_legend,
               edgecolor='None', frameon=False,
               labelspacing=0.2)
    if save:
        assert save_path, "Path to save must be provided!"
        plt.margins(0, 0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()