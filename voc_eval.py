# ----------------------------------------------------------------
# Tensorflow SSD
#
# Written by Geonseok Seo, based on code from SSD by Lukasz Janyst
# ----------------------------------------------------------------
import numpy as np
from ssdutils import jaccard_overlap
from collections import defaultdict

class APCalculate:

    def __init__(self, threshold=0.5, mode='VOC'):
        if mode == 'VOC':
            self.num_cl = 20
        elif mode == 'KITTI':
            self.num_cl = 8
        self.threshold = threshold
        self.gt_class_count = [0]*self.num_cl
        self.pred_class_count = [0]*self.num_cl
        self.detect_boxs_per_class = defaultdict(list)
        self.gt = defaultdict(lambda: defaultdict(list))
        self.match = defaultdict(lambda: defaultdict(list))
        self.AP = [0]*self.num_cl
        self.APs = 0

    def seperate_class(self, gt, gt_label, gt_id, detect_boxs):
        for i, boxes in enumerate(detect_boxs):
            self.pred_class_count[boxes.cl] += 1
            self.detect_boxs_per_class[boxes.cl].append([boxes, gt_id])

        for i, boxes in enumerate(gt):
            if np.count_nonzero(boxes) == 0:
                break
            self.gt_class_count[gt_label[i]] += 1 # class K 에 대한 gt box 개수들 필요
            self.gt[gt_label[i]][gt_id].append(boxes) # class K : gt_label[i], image_name : gt_i

    def compute_ap(self): # gt_boxes_cord, gt_label, bbox_list_nms
        for class_label, value in self.gt.items():
            for index, value2 in value.items():
                self.match[class_label][index] = [0] * len(value2)

        for k in range(self.num_cl): ## k개 class..........
            self.detect_boxs_per_class[k].sort(key=lambda x: x[0].conf, reverse=True)
            #np.argsort(-self.detect_boxs_per_class[k].conf)
            ## detection confidence 큰 순서대로 ............

            tp = np.zeros(self.pred_class_count[k]) ## detection box들 에서 class K인 개수............
            fp = np.zeros(self.pred_class_count[k]) ## detection box들 에서 class K인 개수............
            for i in range(self.pred_class_count[k]): ## detection box들 에서 class K인 개수............
                index = self.detect_boxs_per_class[k][i][1]  # sample_id = 00004.jpg에서 cat 나옴
                box = self.detect_boxs_per_class[k][i][0]

                if index not in self.gt[k].keys(): # gt 00004.jpg에 class K가 없다면
                    fp[i] = 1
                    continue

                gt = self.gt[k][index] # class K, 00004.jpg 에 있는 gt cordinate
                bbox = np.array([box.tl_x, box.br_x, box.tl_y, box.br_y])
                iou = jaccard_overlap(bbox, np.array(gt)) # # of box : 1, # of gt : 10...
                max_index = np.argmax(iou)

                #self.gt[k][index]에 값 없을때도 잇음 -> 아닌듯?..

                if iou[max_index] < self.threshold or self.match[k][index][max_index] == True:
                    fp[i] = 1
                    continue

                tp[i] = 1
                self.match[k][index][max_index] = True

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            recall = tp/self.gt_class_count[k] # detection box들에서  class k 인 개수
            imsi_precision = tp/(tp+fp)
            ap = 0
            for recall_range in np.arange(0, 1.1, 0.1):
                precision = imsi_precision[recall>=recall_range]
                if len(precision) > 0:
                    ap += np.amax(precision) # precision array 의 최댓값을 반환

            ap /= 11. # recall has 11 lines
            self.AP[k] = ap
            self.APs += ap

        self.APs /= self.num_cl

        #return self.APs