# ----------------------------------------------------------------
# Tensorflow SSD
#
# Written by Geonseok Seo, based on code from SSD by Lukasz Janyst
# ----------------------------------------------------------------
import numpy as np
from math import exp
from math import sqrt
import cv2

class BBox(object):
    def __init__(self, cl, conf, tl_y, tl_x, br_y, br_x):
        self.cl = cl
        self.conf = conf
        self.tl_y = tl_y
        self.tl_x = tl_x
        self.br_y = br_y
        self.br_x = br_x

    def __lt__(self, other):
        return self.conf < other.conf

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c) # prevent overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def translate_pred_to_bbox(pred, img_size, num_cl, pred_info):
    dbox_center = np.zeros(((pred.shape[0], pred.shape[1]) + (2,)), dtype=np.float32)
    for y in range(pred.shape[0]):
        for x in range(pred.shape[1]):
            dbox_center[y, x, 0] = (float(x)+0.5) / float(pred.shape[0])
            dbox_center[y, x, 1] = (float(y)+0.5) / float(pred.shape[1])

    bbox_list = list()
    pos_dboxx = np.split(pred,len(pred_info.asp_ratios), -1)

    for k, (pos_dbox, asp_ratio) in enumerate(zip(pos_dboxx, pred_info.asp_ratios)):
        for y in range(pred.shape[0]):
            for x in range(pred.shape[1]):
                pos_dbox_cl = pos_dbox[y][x][:num_cl+1]
                pos_dbox_cl = softmax(pos_dbox_cl)
                pos_dbox_loc = pos_dbox[y][x][num_cl+1:]

                pred_cl = np.argmax(pos_dbox_cl, axis=0)
                if pred_cl == num_cl:#background 제거
                    continue
                pred_conf = pos_dbox_cl[pred_cl]

                if pred_conf < 0.1:#threshold 줌 print 0.01 해보기
                    continue

                dbox_size = []
                if k == len(pred_info.asp_ratios)-1:
                    dbox_size.append(sqrt(pred_info.dbox_scale * pred_info.extra_scale) * sqrt(asp_ratio))
                    dbox_size.append(sqrt(pred_info.dbox_scale * pred_info.extra_scale) / sqrt(asp_ratio))
                else:
                    dbox_size.append(pred_info.dbox_scale * sqrt(asp_ratio))
                    dbox_size.append(pred_info.dbox_scale / sqrt(asp_ratio))

                bbox_center = ((pos_dbox_loc[0] * dbox_size[0]/10.0 + dbox_center[y, x, 0]) * img_size[0],
                               (pos_dbox_loc[1] * dbox_size[1]/10.0 + dbox_center[y, x, 1]) * img_size[1])

                bbox_size = (dbox_size[0] * exp(pos_dbox_loc[2]/5.0) * img_size[0],
                             dbox_size[1] * exp(pos_dbox_loc[3]/5.0) * img_size[1])

                bbox_half_size = (bbox_size[0] * 0.5, bbox_size[1] * 0.5)

                bbox = BBox(pred_cl, pred_conf,
                            tl_x=bbox_center[0] - bbox_half_size[0],
                            tl_y=bbox_center[1] - bbox_half_size[1],
                            br_x=bbox_center[0] + bbox_half_size[0],
                            br_y=bbox_center[1] + bbox_half_size[1])

                if bbox.br_x > 300 :
                    bbox.br_x = 300.
                if bbox.br_y > 300 :
                    bbox.br_y = 300.
                if bbox.tl_x < 0:
                    bbox.tl_x = 0.
                if bbox.tl_y < 0:
                    bbox.tl_y = 0.
                bbox_list.append(bbox)

    return bbox_list

def nms(bbox_list, threshold):

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    conf = []

    for box in bbox_list:
        xmin.append(box.tl_x)
        xmax.append(box.br_x)
        ymin.append(box.tl_y)
        ymax.append(box.br_y)
        conf.append(box.conf)

    xmin = np.array(xmin)
    xmax = np.array(xmax)
    ymin = np.array(ymin)
    ymax = np.array(ymax)
    conf = np.array(conf)

    #---------------------------------------------------------------------------
    # Compute the area of each box and sort the indices by confidence level
    # (lowest confidence first first).
    #---------------------------------------------------------------------------
    area = (xmax-xmin+1) * (ymax-ymin+1)
    idxs = np.argsort(conf)#오름차순
    #idxs = np.arange(len(bbox_list))
    pick = []

    #---------------------------------------------------------------------------
    # Loop until we still have indices to process
    #---------------------------------------------------------------------------
    while len(idxs) > 0:
        #-----------------------------------------------------------------------
        # Grab the last index (ie. the most confident detection), remove it from
        # the list of indices to process, and put it on the list of picks
        #-----------------------------------------------------------------------
        last = idxs.shape[0]-1
        i    = idxs[last]
        idxs = np.delete(idxs, last)
        pick.append(i)# answer

        #-----------------------------------------------------------------------
        # Figure out the intersection with the remaining windows
        #-----------------------------------------------------------------------
        xxmin = np.maximum(xmin[i], xmin[idxs])
        xxmax = np.minimum(xmax[i], xmax[idxs])
        yymin = np.maximum(ymin[i], ymin[idxs])
        yymax = np.minimum(ymax[i], ymax[idxs])

        w = np.maximum(0, xxmax-xxmin+1)# 박스끼리 겹치지 않는 경우 w,h 0으로 만들어 줌
        h = np.maximum(0, yymax-yymin+1)
        intersection = w*h

        #-----------------------------------------------------------------------
        # Compute IOU and suppress indices with IOU higher than a threshold
        #-----------------------------------------------------------------------
        union    = area[i]+area[idxs]-intersection #area[i] = conf가장 큰 박스, area[idxs] = 나머지 박스들
        iou      = intersection/union#--> 뽑은 박스와 나머지 박스들 간의 iou구함
        overlap  = iou > threshold
        suppress = np.nonzero(overlap)[0]#threshold 이상인거 index들 다 뽑음
        idxs     = np.delete(idxs, suppress)

    #---------------------------------------------------------------------------
    # Return the selected boxes
    #---------------------------------------------------------------------------
    selected = []
    for i in pick:
        selected.append(bbox_list[i])

    return selected

def draw_box(img, box, text, color):
    xmin, xmax, ymin, ymax = int(box.tl_x), int(box.br_x), int(box.tl_y), int(box.br_y)
    box_img = img
    cv2.rectangle(box_img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(box_img, (xmin-1, ymin), (xmax+1, ymin-20), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(box_img, text, (xmin+5, ymin-5), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
