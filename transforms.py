# ----------------------------------------------------------------
# Tensorflow SSD
#
# Written by Geonseok Seo, based on code from SSD by Lukasz Janyst
# ----------------------------------------------------------------
import random
import numpy as np
from PIL import Image, ImageEnhance
import os
from ssdutils import get_anchors_for_preset, anchors2array, box2array
from ssdutils import compute_overlap, process_overlap
from utils import Size, Box, Sample, abs2prop, Point
from utils import prop2abs
from math import sqrt
import lxml.etree
from collections import namedtuple

Anchor = namedtuple('Anchor', ['center', 'size'])

class Transform:
    def __init__(self, **kwargs):
        for arg, val in kwargs.items(): #arg, val : key, value in dictionary
            setattr(self, arg, val) #=> self.arg = val로 초기화
        self.initialized = False

def transform_box(box, orig_size, new_size, h_off, w_off):
    #---------------------------------------------------------------------------
    # Compute the new coordinates of the box
    #---------------------------------------------------------------------------
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, orig_size)
    xmin += w_off
    xmax += w_off
    ymin += h_off
    ymax += h_off

    #---------------------------------------------------------------------------
    # Check if the center falls within the image
    #---------------------------------------------------------------------------
    width = xmax - xmin
    height = ymax - ymin
    new_cx = xmin + int(width/2)
    new_cy = ymin + int(height/2)
    if new_cx < 0 or new_cx >= new_size.w:
        return None
    if new_cy < 0 or new_cy >= new_size.h:
        return None

    center, size = abs2prop(xmin, xmax, ymin, ymax, new_size)

    return Box(box.label, box.labelid, center, size)

#-------------------------------------------------------------------------------
def transform_gt(gt, new_size, h_off, w_off):
    boxes = []
    for box in gt.boxes:
        box = transform_box(box, gt.imgsize, new_size, h_off, w_off)
        if box is None:
            continue
        boxes.append(box)
    return Sample(gt.filename, boxes, new_size)

class SamplePickerTransform(Transform):
    """
    Run a bunch of sample transforms and return one of the produced samples
    Parameters: samplers
    """
    def __call__(self, data, label, gt):
        samples = []
        for sampler in self.samplers:
            sample = sampler(data, label, gt)
            if sample is not None:
                samples.append(sample)
        return random.choice(samples)

class SamplerTransform(Transform):
    """
    Sample a fraction of the image according to given parameters
    Params: min_scale, max_scale, min_aspect_ratio, max_aspect_ratio,
            min_jaccard_overlap
    """
    def __call__(self, data, label, gt):
        #-----------------------------------------------------------------------
        # Check whether to sample or not
        #-----------------------------------------------------------------------
        if not self.sample:
            return data, label, gt

        #-----------------------------------------------------------------------
        # Retry sampling a couple of times
        #-----------------------------------------------------------------------
        source_boxes = anchors2array(gt.boxes, gt.imgsize) # get abs value(xmin,xmax,ymin,ymax)
        box = None
        box_arr = None
        for _ in range(self.max_trials):
            #-------------------------------------------------------------------
            # Sample a bounding box
            #-------------------------------------------------------------------
            # min_scale=0.3, max_scale=1.0,
            # min_aspect_ratio=0.5, max_aspect_ratio=2.0,
            scale = random.uniform(self.min_scale, self.max_scale)
            aspect_ratio = random.uniform(self.min_aspect_ratio,
                                          self.max_aspect_ratio)

            # make sure width and height will not be larger than 1
            aspect_ratio = max(aspect_ratio, scale**2)
            aspect_ratio = min(aspect_ratio, 1/(scale**2))

            width = scale*sqrt(aspect_ratio)
            height = scale/sqrt(aspect_ratio)
            cx = 0.5*width + random.uniform(0, 1-width)
            cy = 0.5*height + random.uniform(0, 1-height)
            center = Point(cx, cy)
            size = Size(width, height)

            #-------------------------------------------------------------------
            # Check if the box satisfies the jaccard overlap constraint
            #-------------------------------------------------------------------
            box_arr = np.array(prop2abs(center, size, gt.imgsize))
            overlap = compute_overlap(box_arr, source_boxes, 0)
            if overlap.best and overlap.best.score >= self.min_jaccard_overlap:
                box = Box(None, None, center, size)
                break

        if box is None:
            return None

        #-----------------------------------------------------------------------
        # Crop the box and adjust the ground truth
        #-----------------------------------------------------------------------
        new_size = Size(box_arr[1]-box_arr[0], box_arr[3]-box_arr[2])
        w_off = -box_arr[0]
        h_off = -box_arr[2]

        data = data.crop((box_arr[0], box_arr[2], box_arr[1], box_arr[3])) # gt와 gt근처의 가상의 anchor를 잡은뒤, iou 0.1, 0.3, 0.5, 0.7, 0.9 이상이면 그 가상의 anchor부분을 crop하고 gt 좌표도 같이 변경
        # 즉 일부로 gt와 iou 연관있는 그림부분을 crop함
        gt = transform_gt(gt, new_size, h_off, w_off)

        return data, label, gt # change gt and image_size

class BrightnessTransform(Transform):
    def __call__(self, data, label, gt):
        delta = random.uniform(self.lower, self.upper)
        data = ImageEnhance.Brightness(data).enhance(delta)
        return data, label, gt

class ContrastTransform(Transform):
    def __call__(self, data, label, gt):
        delta = random.uniform(self.lower, self.upper)
        data = ImageEnhance.Contrast(data).enhance(delta)
        return data, label, gt

class HueTransform(Transform):
    def __call__(self, data, label, gt):
        #RGB2HSV
        h, s, v = data.convert('HSV').split()
        np_h = np.array(h, dtype=np.uint8)
          # uint8 addition take cares of rotation across boundaries
        delta = random.uniform(self.lower, self.upper)
        with np.errstate(over='ignore'):
            np_h += np.uint8(delta * 255)
        h = Image.fromarray(np_h, 'L')
        img = Image.merge('HSV', (h, s, v)).convert('RGB')
        #HSV2RGB
        return img, label, gt

class SaturationTransform(Transform):
    def __call__(self, data, label, gt):
        delta = random.uniform(self.lower, self.upper)
        data = ImageEnhance.Color(data).enhance(delta)
        return data, label, gt

class ComposeTransform(Transform):
     """
     Call a bunch of transforms serially
     Parameters: transforms
     """
     def __call__(self, data, label, gt):
         args = (data, label, gt)
         for t in self.transforms:
             args = t(*args)
         return args

class TransformPickerTransform(Transform):
     """
     Call a randomly chosen transform from the list
     Parameters: transforms
     """
     def __call__(self, data, label, gt):
         pick = random.randint(0, len(self.transforms)-1)
         return self.transforms[pick](data, label, gt)

class ImageLoaderTransform(Transform):
    """
    Load image
    """
    def __call__(self, fname):
        with Image.open(fname) as img:
            img = img.convert('RGB')
        img_size = Size(img.size[0], img.size[1])

        if self.dataset == 'VOC':
            #fname : '~/VOC/JPEGImages/002222.jpg'
            with open(os.path.join(fname[:-21], 'Annotations', fname[-10:-4]+'.xml'), 'r') as f:
                doc = lxml.etree.parse(f)
                boxes = []
                objects = doc.xpath('/annotation/object')

                for obj in objects:
                    label = obj.xpath('name')[0].text
                    xmin = float(obj.xpath('bndbox/xmin')[0].text)
                    xmax = float(obj.xpath('bndbox/xmax')[0].text)
                    ymin = float(obj.xpath('bndbox/ymin')[0].text)
                    ymax = float(obj.xpath('bndbox/ymax')[0].text)

                    labels = self.label2idx[label]

                    center, size = abs2prop(xmin, xmax, ymin, ymax, img_size) # make gt to 0~1 important!!!!!!!!!!!!!!
                    box = Box(obj, labels, center, size)
                    boxes.append(box)

        elif self.dataset == 'KITTI':
            with open(os.path.join(fname[:-18], 'label_2', fname[-10:-4]+'.txt'), 'r') as fp:
                objs = [line.split(' ') for line in fp.readlines()]
            boxes = []
            for obj in objs:
                if not obj[0] == 'DontCare':
                    xmin = float(obj[4])
                    ymin = float(obj[5])
                    xmax = float(obj[6])
                    ymax = float(obj[7])
                    label = self.label2idx[obj[0]]
                    center, size = abs2prop(xmin, xmax, ymin, ymax, img_size) # Convert the absolute min-max box bound to proportional center-width bounds
                    # (/300 , /1300) -> (0~1, 0~1)
                    box = Box(obj[0], label, center, size)
                    boxes.append(box)

        sample = Sample(fname, boxes, img_size)
        return img, labels, sample

class ExpandTransform(Transform):
    """
    Expand the image and fill the empty space with the mean value
    Parameters: max_ratio, mean_value
    """
    def __call__(self, data, label, gt):
        #-----------------------------------------------------------------------
        # Calculate sizes and offsets
        #-----------------------------------------------------------------------
        ratio = random.uniform(1, self.max_ratio)       # 2

        orig_size = gt.imgsize                          # 400,300
        new_size = Size(int(orig_size.w*ratio), int(orig_size.h*ratio)) # 800, 600
        h_off = random.randint(0, new_size.h-orig_size.h) # 0 ~ (800-400) => 100
        w_off = random.randint(0, new_size.w-orig_size.w) # 0 ~ (600-300) => 50

        #-----------------------------------------------------------------------
        # Create the new image and place the input image in it
        #-----------------------------------------------------------------------
        imsi = Image.new('RGB', (new_size.w, new_size.h), self.mean_value)
        imsi.paste(data, (w_off, h_off, w_off+orig_size.w, h_off+orig_size.h))

        #-----------------------------------------------------------------------
        # Transform the ground truth
        #-----------------------------------------------------------------------
        gt = transform_gt(gt, new_size, h_off, w_off)

        return imsi, label, gt

class ResizeTransform(Transform):
    """
    Resize image
    Args:
        width, height
    """
    def __call__(self, data, label, gt):
        alg = random.choice(self.algorithms)
        resized = data.resize((self.width, self.height), alg)
        resized = np.array(resized, dtype='float32')

        return resized, label, gt

class HorizontalFlipTransform(Transform):
    """
    Horizontally flip the image
    """
    def __call__(self, data, label, gt):
        flip = data.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = []
        for box in gt.boxes:
            center = Point(1-box.center.x, box.center.y)
            box = Box(box.label, box.labelid, center, box.size)
            boxes.append(box)
        gt = Sample(gt.filename, boxes, gt.imgsize)
        return flip, label, gt

class RandomTransform(Transform):
    """
    Call another transform with a given probability
    Parameters: prob, transform
    """
    def __call__(self, data, label, gt):
        p = random.uniform(0, 1)
        if p < self.prob:
            return self.transform(data, label, gt)
        return data, label, gt

class ReorderChannelsTransform(Transform):
     """
     Reorder Image Channels
     """
     def __call__(self, data, label, gt):
         r, g, b = data.split()
         p = random.uniform(0,6)
         if p>=0 and p<1:
             img = Image.merge('RGB', (r, b, g))
         elif p>=1 and p<2:
             img = Image.merge('RGB', (g, r, b))
         elif p>=2 and p<3:
             img = Image.merge('RGB', (g, b, r))
         elif p>=3 and p<4:
             img = Image.merge('RGB', (b, r, g))
         elif p>=4 and p<5:
             img = Image.merge('RGB', (b, g, r))
         else:
             img = data
         return img, label, gt

class LabelTransform(Transform):
    """
    Label preprocess
    Args:
        preset, num_classes
    """
    def initialize(self):
        self.anchors = get_anchors_for_preset(self.preset) # 0~1
        self.vheight = len(self.anchors)
        self.vwidth = self.num_classes+5 # background class + location offsets
        self.img_size = Size(300, 300)
        self.anchors_arr = anchors2array(self.anchors, self.img_size) # anchors_arr = (0~300, 0~300)
        self.initialized = True

    def __call__(self, data, label, gt):
        if not self.initialized:
            self.initialize()
        vec = np.zeros((self.vheight, self.vwidth), dtype=np.float32)
        overlaps = {}
        for box in gt.boxes:
            # box = 0~1
            box_arr = box2array(box, self.img_size) # Convert proportional center-width bounds to absolute min-max bounds
            # box_arr = (0~300, 0~300)
            # gt_box xmin, xmax, ymin, ymax
            overlaps[box] = compute_overlap(box_arr, self.anchors_arr, 0.4)

        vec[:, self.num_classes] = 1.    # background
        vec[:, self.num_classes+1] = 0.  # x offset
        vec[:, self.num_classes+2] = 0.  # y offset
        vec[:, self.num_classes+3] = 0.  # log width scale
        vec[:, self.num_classes+4] = 0.  # log height scale

        matches = {}
        for box in gt.boxes:
            for overlap in overlaps[box].good:
                anchor = self.anchors[overlap.idx]
                process_overlap(overlap, box, anchor, matches, self.num_classes, vec)

        matches = {}
        for box in gt.boxes:
            overlap = overlaps[box].best
            if not overlap:
                continue
            anchor = self.anchors[overlap.idx]
            process_overlap(overlap, box, anchor, matches, self.num_classes, vec)

        return data, vec, gt.filename

class LabelTransform_valid(Transform):

    def initialize(self):
        self.img_size = Size(300, 300)
        self.initialized = True

    def __call__(self, data, label, gt):
        if not self.initialized:
            self.initialize()

        box_cordinate = list()
        gt_label = list()
        for boxes in gt.boxes:
            box = Anchor(Point(boxes[2][0], boxes[2][1]), Size(boxes[3][0], boxes[3][1]))
            box_cordinate.append(box)
            gt_label.append(boxes[1])
        box_arr = anchors2array(box_cordinate, self.img_size) # gt h,w over 300?
        vec = np.zeros((100-len(box_arr), 4), dtype=np.double)
        label = np.zeros(100-len(box_arr), dtype=np.int64)
        vec = np.r_[box_arr, vec]
        label = np.r_[gt_label, label]

        return data, gt.filename, label, tuple(gt.imgsize), vec#, gt[1]