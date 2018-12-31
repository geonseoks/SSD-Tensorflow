# ----------------------------------------------------------------
# Tensorflow SSD
#
# Written by Geonseok Seo, based on code from SSD by Lukasz Janyst
# ----------------------------------------------------------------
import os.path as osp
import tensorflow as tf
import os
import bbox
import cv2
import numpy as np
from voc_eval import APCalculate
import argparse
from tqdm import tqdm
from ssd_network import SSD
from dataset import Dataset
from utils import Label
from collections import defaultdict

label_defs = {'VOC':(
    Label('dog', (30, 170, 250)),
    Label('sofa', (142, 0, 0)),
    Label('chair', (153, 153, 190)),
    Label('boat', (128,  64, 128)),
    Label('bicycle', (0, 74, 111)),
    Label('car', ( 70,  70,  70)),
    Label('pottedplant', (180, 130, 70)),
    Label('horse', (0, 220, 220)),
    Label('cow', (90, 120, 150)),
    Label('person', ( 52, 151,  52)),
    Label('diningtable', (153, 153, 153)),
    Label('bottle', (232, 35, 244)),
    Label('cat', (156, 102, 102)),
    Label('bird', (81, 0, 81)),
    Label('train', (230, 0, 0)),
    Label('motorbike', (35, 142, 107)),
    Label('tvmonitor', (32, 11, 119)),
    Label('aeroplane', (0,     0,   0)),
    Label('bus', (140, 150, 230)),
    Label('sheep', (60, 20, 220))),
    'KITTI':(
    Label('car', (0, 128, 0)),
    Label('truck', (140, 150, 230)),
    Label('misc', (232, 35, 244)),
    Label('van', (128, 64, 128)),
    Label('pedestrian', (0, 74, 111)),
    Label('cyclist', (81, 0, 81)),
    Label('tram', (70, 70, 70)),
    Label('person_sitting', (156, 102, 102)),
    Label('background', (153, 153, 190))
    )
}

def inference(args):
    model_file = osp.join(
        args.output_dir, args.ex_dir,
        'epoch_{:d}'.format(args.epoch_num) + '.ckpt')
    if not os.path.exists(os.path.join(args.output_dir, args.ex_dir)):
        os.makedirs(os.path.join(args.output_dir, args.ex_dir))
    if not os.path.exists(os.path.join(args.output_dir, args.ex_dir, 'eval')):
        os.makedirs(os.path.join(args.output_dir, args.ex_dir, 'eval'))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)    #선택된 device 사용할 수 없을때 다른 장치를 찾기를 원할때
    tfconfig.gpu_options.allow_growth = True                #실행 과정에서 요구되는 만큼의 GPU 메모리만 할당하게 함
    colors = [l.color for l in label_defs[args.test_set]]
    AP = APCalculate(args.iou_threshold, args.test_set)

    if args.batch_size % args.num_gpus is not 0:
        print('please enter batch_size and num_gpus again(could not divide)')
        exit()
    args.batch_size = args.batch_size//args.num_gpus

    with tf.device('/cpu:0'):
        dataset = Dataset(os.path.dirname(__file__), args.batch_size, 'test', args.num_gpus)
        num_valid = dataset.valid_train
        args.num_cl = dataset.num_classes
        net = SSD(args, args.backbone_dir)
        for i in range(args.num_gpus):
            with tf.device('gpu:' + str(i)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(i > 0)):
                    net.create_network(i * args.batch_size, (i + 1) * args.batch_size)
                    saver = tf.train.Saver()

                # if i == 0:
                # Evaluate model (with test logits, for dropout to be disabled)
                #    correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(_y, 1))
                #    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        total_batch = num_valid // (args.batch_size * args.num_gpus)

        tfconfig = tf.ConfigProto(allow_soft_placement=True)  # 선택된 device 사용할 수 없을때 다른 장치를 찾기를 원할때
        tfconfig.gpu_options.allow_growth = True  # 실행 과정에서 요구되는 만큼의 GPU 메모리만 할당하게 함
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4 # GPU 몇프로 사용할 것인지 0~1

        sess = tf.Session(config=tfconfig)
        # Initialize the variables (i.e. assign their default value)

        init = tf.global_variables_initializer()
        sess.run(init)

        iterator = dataset.get_iterator()
        next_element = iterator.get_next()

        saver.restore(sess, model_file)
        sess.run(iterator.initializer)
        end_flag = 0

        for iter, _ in enumerate(tqdm(range(total_batch+1), desc='# ' + args.test_set + ' evaluation processing')):
            # batch32 , num_gpu = 2 -> 32*2 = 64 data씩 가져오게..
            # TODO: get_batch
            test_batch = dict()
            output_data = sess.run(next_element)
            if iter == 0:
                input_first = output_data
            if iter == total_batch:
                if num_valid % (args.batch_size * args.num_gpus) == 0:
                    break
                else:
                    test_batch['image'] = np.append(output_data[0], input_first[0][:args.batch_size*args.num_gpus-num_valid % (args.batch_size * args.num_gpus)], axis=0)
            if iter < total_batch:
                test_batch['image'] = output_data[0]  # 0~255

            feed_dict = {net.image: test_batch['image']}
            run_list = [tf.get_collection('bbox_pred1'), tf.get_collection('bbox_pred2'),
                        tf.get_collection('bbox_pred3'), tf.get_collection('bbox_pred4'),
                        tf.get_collection('bbox_pred5'), tf.get_collection('bbox_pred6')]
            pred1, pred2, pred3, pred4, pred5, pred6 = sess.run(run_list, feed_dict=feed_dict)

            for j in range(args.num_gpus):
                for l in range(args.batch_size):
                    pred = [pred1[j][l], pred2[j][l], pred3[j][l], pred4[j][l], pred5[j][l], pred6[j][l]]
                    bbox_list = list()

                    for k, prediction in enumerate(pred):
                        bbox_list += bbox.translate_pred_to_bbox(prediction, (args.img_h, args.img_w), args.num_cl, net.pred_infos[k])

                    bbox_list.sort(reverse=True, key=lambda student: student.conf)

                    top_k = 200
                    if len(bbox_list) < top_k:
                        bbox_list = bbox_list[:len(bbox_list)]
                    else:
                        bbox_list = bbox_list[:top_k]

                    bbox_list2 = defaultdict(list)
                    [bbox_list2[boxes.cl].append(boxes) for boxes in bbox_list]

                    bbox_list_nms = list()
                    for k in range(args.num_cl):
                        bbox_list_nms += bbox.nms(bbox_list2[k], args.nms_threshold)

                    img = test_batch['image'][j*args.batch_size+l].astype(np.uint8)
                    test_box_img = np.copy(img)
                    [bbox.draw_box(test_box_img, box, label_defs[args.test_set][box.cl].name, colors[box.cl]) for box in bbox_list_nms]
                    test_box_img = cv2.cvtColor(test_box_img, cv2.COLOR_BGR2RGB)
                    if args.show_img == True:
                        cv2.imshow('box_img', test_box_img)
                        cv2.waitKey(0)

                    gt_id = output_data[1][j*args.batch_size+l][len(output_data[1][j*args.batch_size+l])-10:len(output_data[1][j*args.batch_size+l])-4].decode("utf-8")
                    gt_label = output_data[2][j*args.batch_size+l]
                    gt_boxes_cord = output_data[4][j*args.batch_size+l] # read gt_label and boxes_cordinates 100 size

                    AP.seperate_class(gt_boxes_cord, gt_label, gt_id, bbox_list_nms)
                    if args.save_img == True:
                        cv2.imwrite(os.path.join(args.output_dir, args.ex_dir, 'eval/') + gt_id + '.png', test_box_img)
                    if iter == total_batch and j * args.batch_size + l + 1 == num_valid % (args.batch_size * args.num_gpus):
                        end_flag = 1
                        break
                if end_flag == 1:
                    break

        AP.compute_ap()
        print('\n------ VOC07 metric ------')
        for i in range(args.num_cl):
            print(label_defs[args.test_set][i].name, "AP : {:.2f}".format(AP.AP[i]))

        print('mAP : ', AP.APs)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_w', '--img_w', type=int, default=300, help='Width of the image.')
    parser.add_argument('-img_h', '--img_h', type=int, default=300, help='Height of the image.')
    parser.add_argument('-max_e', '--epoch_num', type=int, default=300, help='load training epoch ckpt file')
    parser.add_argument('-num_classes', '--num_cl', type=int, default=0, help='number of classes')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch_size')
    parser.add_argument('-log_dir', '--log_dir', default='./log_dir', help='log_dir')
    parser.add_argument('-outdir','--output_dir', default='/home/gs/SSD-Tensorflow/output', help='output_dir')
    parser.add_argument('-experimentdir', '--ex_dir', default='model_dump', help='experiment_dir')
    parser.add_argument('-nmsthres','--nms_threshold', type=float, default=0.4, help='nms_threshold')
    parser.add_argument('-iouthres', '--iou_threshold', type=float, default=0.5, help='iou_threshold')
    parser.add_argument('-num_gpus','--num_gpus', type=int, default=2, help='number of gpus')
    parser.add_argument('-data_set', '--test_set', default='VOC', type=str, help='training_set : VOC, KITTI')
    parser.add_argument('-show_img', '--show_img', default=False, type=bool, help='Show the test_box_image')
    parser.add_argument('-save_img', '--save_img', default=False, type=bool, help='Save the test_box_image')
    parser.add_argument('-backbonedir', '--backbone_dir', default='/home/gs/SSD-Tensorflow/vgg16.npy', help='Backbone dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    inference(args)
