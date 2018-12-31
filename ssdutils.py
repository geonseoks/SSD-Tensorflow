# ----------------------------------------------------------------
# Tensorflow SSD
#
# Written by Geonseok Seo, based on code from SSD by Lukasz Janyst
# ----------------------------------------------------------------
import numpy as np
from collections import namedtuple
from math import sqrt, log

from utils import Size, Point, Score, Overlap
from utils import prop2abs

SSDMap = namedtuple('SSDMap', ['size', 'scale', 'aspect_ratios'])
SSDPreset = namedtuple('SSDPreset', ['name', 'image_size', 'maps',
                                     'extra_scale', 'num_anchors'])

Anchor = namedtuple('Anchor', ['center', 'size', 'x', 'y', 'scale', 'map'])

SSD_PRESETS = {
    'vgg300': SSDPreset(name = 'vgg300',
                        image_size = Size(300, 300),
                        maps = [
                            SSDMap(Size(38, 38), 0.1,   [2.0, 0.5]),
                            SSDMap(Size(19, 19), 0.2,   [2.0, 3.0, 0.5, 1./3.]),
                            SSDMap(Size(10, 10), 0.375, [2.0, 3.0, 0.5, 1./3.]),
                            SSDMap(Size( 5,  5), 0.55,  [2.0, 3.0, 0.5, 1./3.]),
                            SSDMap(Size( 3,  3), 0.725, [2.0, 0.5]),
                            SSDMap(Size( 1,  1), 0.9,   [2.0, 0.5])
                        ],
                        extra_scale = 1.075,
                        num_anchors = 8732),
    'vgg512': SSDPreset(name = 'vgg512',
                        image_size = Size(512, 512),
                        maps = [
                            SSDMap(Size(64, 64), 0.07, [2., 0.5]),
                            SSDMap(Size(32, 32), 0.15, [2., 3., 0.5, 1./3.]),
                            SSDMap(Size(16, 16), 0.3,  [2., 3., 0.5, 1./3.]),
                            SSDMap(Size( 8,  8), 0.45, [2., 3., 0.5, 1./3.]),
                            SSDMap(Size( 4,  4), 0.6,  [2., 3., 0.5, 1./3.]),
                            SSDMap(Size( 2,  2), 0.75, [2., 0.5]),
                            SSDMap(Size( 1,  1), 0.9,  [2., 0.5])
                        ],
                        extra_scale = 105,
                        num_anchors = 24564)
}

def get_preset_by_name(pname):
    if not pname in SSD_PRESETS:
        raise RuntimeError('No such preset: '+pname)
    return SSD_PRESETS[pname]

def get_anchors_for_preset(preset):
    """
    Compute the default (anchor) boxes for the given SSD preset
    """
    #---------------------------------------------------------------------------
    # Compute the width and heights of the anchor boxes for every scale
    #---------------------------------------------------------------------------
    box_sizes = []

    for i in range(len(preset.maps)):
        map_params = preset.maps[i]
        s = map_params.scale
        aspect_ratios = [1] + map_params.aspect_ratios
        aspect_ratios = list(map(lambda x: sqrt(x), aspect_ratios))

        sizes = []
        for ratio in aspect_ratios:
            w = s * ratio
            h = s / ratio
            sizes.append((w, h))
        if i < len(preset.maps)-1:
            s_prime = sqrt(s*preset.maps[i+1].scale)
        else:
            s_prime = sqrt(s*preset.extra_scale)
        sizes.append((s_prime, s_prime))
        box_sizes.append(sizes)

    #---------------------------------------------------------------------------
    # Compute the actual boxes for every scale and feature map
    #---------------------------------------------------------------------------
    anchors = []

    for k in range(len(preset.maps)):
        fk = preset.maps[k].size[0]
        for size in box_sizes[k]:
            for j in range(fk):
                y = (j+0.5)/float(fk)
                for i in range(fk):
                    x = (i+0.5)/float(fk)
                    box = Anchor(Point(x, y), Size(size[0], size[1]),
                                 i, j, s, k)
                    anchors.append(box)
    return anchors

def anchors2array(anchors, img_size):
    """
    Computes a numpy array out of absolute anchor params (img_size is needed
    as a reference)
    """
    arr = np.zeros((len(anchors), 4))
    #print(len(anchors))
    for i in range(len(anchors)):
        anchor = anchors[i]
        xmin, xmax, ymin, ymax = prop2abs(anchor.center, anchor.size, img_size)
        arr[i] = np.array([xmin, xmax, ymin, ymax])

    return arr

def box2array(box, img_size):
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    return np.array([xmin, xmax, ymin, ymax])

def jaccard_overlap(box_arr, anchors_arr):
    areaa = (anchors_arr[:, 1]-anchors_arr[:, 0]+1) * \
            (anchors_arr[:, 3]-anchors_arr[:, 2]+1)
    areab = (box_arr[1]-box_arr[0]+1) * (box_arr[3]-box_arr[2]+1)

    xxmin = np.maximum(box_arr[0], anchors_arr[:, 0])
    xxmax = np.minimum(box_arr[1], anchors_arr[:, 1])
    yymin = np.maximum(box_arr[2], anchors_arr[:, 2])
    yymax = np.minimum(box_arr[3], anchors_arr[:, 3])

    w = np.maximum(0, xxmax-xxmin+1)
    h = np.maximum(0, yymax-yymin+1)
    intersection = w*h
    union = areab+areaa-intersection
    return intersection/union

def compute_overlap(box_arr, anchors_arr, threshold):
    iou = jaccard_overlap(box_arr, anchors_arr)
    overlap = iou > threshold

    good_idxs = np.nonzero(overlap)[0]
    best_idx  = np.argmax(iou)
    best = None
    good = []

    if iou[best_idx] > threshold:
        best = Score(best_idx, iou[best_idx])

    for idx in good_idxs:
        good.append(Score(idx, iou[idx]))

    return Overlap(best, good)

def compute_location(box, anchor):
    arr = np.zeros((4))
    arr[0] = (box.center.x-anchor.center.x)/anchor.size.w*10
    arr[1] = (box.center.y-anchor.center.y)/anchor.size.h*10
    arr[2] = log(box.size.w/anchor.size.w)*5
    arr[3] = log(box.size.h/anchor.size.h)*5
    return arr

def process_overlap(overlap, box, anchor, matches, num_classes, vec):
    if overlap.idx in matches and matches[overlap.idx] >= overlap.score:
        return

    matches[overlap.idx] = overlap.score
    vec[overlap.idx, 0:num_classes+1] = 0
    vec[overlap.idx, box.labelid]     = 1

    vec[overlap.idx, num_classes+1:]  = compute_location(box, anchor)
