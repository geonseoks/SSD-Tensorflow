# ----------------------------------------------------------------
# Tensorflow SSD
#
# Written by Geonseok Seo, based on code from SSD by Lukasz Janyst
# ----------------------------------------------------------------
import tensorflow as tf
import argparse, os
import numpy as np
from ssd_network import SSD
from tensorboardX import SummaryWriter
from dataset import Dataset
from collections import defaultdict

VGG_MEAN = [103.939, 116.779, 123.68]

import bbox
import cv2
import inference

def snapshot(sess,saver,epoch, experiment_name):
    filename = 'epoch_{:d}'.format(epoch)+'.ckpt'
    model_dump_dir = os.path.join(args.output_dir, experiment_name)
    if not os.path.exists(model_dump_dir):
        os.makedirs(model_dump_dir)
    filename = os.path.join(model_dump_dir,filename)
    saver.save(sess,filename)
    print('Wrote snapshot to: {:s}'.format(filename))


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    train_log_dir = os.path.join(args.log_dir, 'train')
    valid_log_dir = os.path.join(args.log_dir, 'valid')
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.path.exists(valid_log_dir):
        os.makedirs(valid_log_dir)
    if not os.path.exists(os.path.join(args.output_dir, args.ex_dir)):
        os.makedirs(os.path.join(args.output_dir, args.ex_dir))
    if not os.path.exists(os.path.join(args.output_dir, args.ex_dir, 'image')):
        os.makedirs(os.path.join(args.output_dir, args.ex_dir, 'image'))
    train_log_writer = SummaryWriter(train_log_dir)
    # valid_log_writer = SummaryWriter(valid_log_dir)

    if args.batch_size % args.num_gpus is not 0:
        print('please enter batch_size and num_gpus again(could not divide)')
        exit()
    args.batch_size = args.batch_size//args.num_gpus

    with tf.device('/cpu:0'):
        tower_grads = []
        loss_tmp = []
        cls_loss_tmp = []
        loc_loss_tmp = []
        regul_loss_tmp = []

        learning_rate = tf.placeholder(tf.float32, shape=[])
        optim = tf.train.MomentumOptimizer(learning_rate, args.momentum)
        dataset = Dataset(os.path.dirname(__file__), args.batch_size, 'train', args.num_gpus)
        num_trains = dataset.num_train
        args.num_cl = dataset.num_classes
        net = SSD(args, args.backbone_dir)
        for i in range(args.num_gpus):
            with tf.device('gpu:' + str(i)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(i > 0)):
                    net.create_network(i * args.batch_size, (i+1) * args.batch_size)
                    cls_loss, loc_loss, regul_loss, total_loss = net.create_basic_loss(i * args.batch_size, (i+1) * args.batch_size)

                    loss_tmp.append(total_loss)
                    loc_loss_tmp.append(loc_loss)
                    cls_loss_tmp.append(cls_loss)
                    regul_loss_tmp.append(regul_loss)
                    # train_op = optim.minimize(total_loss)

                    grads = optim.compute_gradients(total_loss) # This is the first part of minimize() //  gradient 값을 계산한 후,
                    tower_grads.append(grads)

        total_batch = num_trains // (args.batch_size * args.num_gpus)
        tower_grads = average_gradients(tower_grads)
        train_op = optim.apply_gradients(tower_grads) # This is the second part of minimize() // learning rate를 곱하여 기존 parameter에서 뺀 값으로 업데이트 시키도록

        total_losss = tf.reduce_mean(loss_tmp)
        global_step = tf.Variable(0, trainable=False)
        #train_op = optim.minimize(total_loss, global_step, colocate_gradients_with_ops=True)
        cls_losss = tf.reduce_mean(cls_loss_tmp)
        loc_losss = tf.reduce_mean(loc_loss_tmp)
        regul_losss = tf.reduce_mean(regul_loss_tmp) # same as tf.add_n(tf.get_collection('regul_loss'))
        tfconfig = tf.ConfigProto(allow_soft_placement=True)  # 선택된 device 사용할 수 없을때 다른 장치를 찾기를 원할때
        tfconfig.gpu_options.allow_growth = True  # 실행 과정에서 요구되는 만큼의 GPU 메모리만 할당하게 함
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4 # GPU 몇프로 사용할 것인지 0~1

        sess = tf.Session(config=tfconfig)
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        sess.run(init)
        global_iter = 0
        lr = args.lr

        iterator = dataset.get_iterator()
        next_element = iterator.get_next()
        for epoch in range(args.max_epoch):
            sess.run(iterator.initializer)
            for iter in range(total_batch+1):  # ex)total_image = 10000, batch = 10 -> total_batch = 10000/10
                # batch32 , num_gpu = 2 -> 32*2 = 64 data씩 가져오게..
                # print(objgraph.show_growth())
                # TODO: get_batch
                input_data = sess.run(next_element)
                train_batch = dict()
                if iter == 0:
                    input_first = input_data
                if iter == total_batch:
                    if num_trains % (args.batch_size * args.num_gpus) == 0:
                        break
                    else:
                        train_batch['image'] = np.append(input_data[0], input_first[0][:args.batch_size*args.num_gpus-num_trains % (args.batch_size * args.num_gpus)], axis=0)
                        #train_list = np.append(input_data[2], input_first[2][:args.batch_size-num_trains % (args.batch_size * args.num_gpus)], axis=0)
                        imsis = np.append(input_data[1], input_first[1][:args.batch_size*args.num_gpus-num_trains % (args.batch_size * args.num_gpus)], axis=0)
                        train_batch['gt_cl'] = imsis[:, :, :args.num_cl + 1]
                        train_batch['gt_loc'] = imsis[:, :, args.num_cl + 1:]

                if iter < total_batch:
                    train_batch['image'] = input_data[0]  # 0~255
                    train_batch['gt_cl'] = input_data[1][:, :, :args.num_cl + 1]  # (32, 8732, 9)
                    train_batch['gt_loc'] = input_data[1][:, :, args.num_cl + 1:]  # (32, 8732, 4)
                    train_list = input_data[2]

                #print(train_list)
                feed_dict = {learning_rate: lr,
                             net.image: train_batch['image'],
                             net.gt_loc: train_batch['gt_loc'],
                             net.gt_cl: train_batch['gt_cl']}
                run_list = [cls_losss, loc_losss, regul_losss, total_losss, train_op] # 우리가 궁금한것 + gradient 업데이트값(train_op)
                all_cls_loss, all_loc_loss, all_regul_loss, all_total_loss, _ = sess.run(run_list, feed_dict=feed_dict)

                print("Epoch %d/%d, Iter: %d/%d" % (epoch, args.max_epoch, iter, total_batch+1))
                print("\tClass loss: %f\n\tLoc loss: %f\n\tRegul loss: %f" % (all_cls_loss, all_loc_loss, all_regul_loss))
                print("\tTotal Loss : %f" % (all_total_loss))
                train_log_writer.add_scalar('cl_loss', all_cls_loss, global_iter)
                train_log_writer.add_scalar('loc_loss', all_loc_loss, global_iter)
                train_log_writer.add_scalar('total_loss', all_total_loss, global_iter)

                if global_iter == 40000:
                    lr *= 0.1
                if global_iter == 50000:
                    lr *= 0.1
                global_iter += 1

                if iter == total_batch and epoch % 10 == 0 and args.save_img == True:
                    feed_dict = {net.image: train_batch['image']}
                    run_list = [tf.get_collection('bbox_pred1'), tf.get_collection('bbox_pred2'),
                                tf.get_collection('bbox_pred3'), tf.get_collection('bbox_pred4'),
                                tf.get_collection('bbox_pred5'), tf.get_collection('bbox_pred6')]
                    pred1, pred2, pred3, pred4, pred5, pred6 = sess.run(run_list, feed_dict=feed_dict)
                    for j in range(args.num_gpus):
                        for l in range(args.batch_size):
                            pred = [pred1[j][l], pred2[j][l], pred3[j][l], pred4[j][l], pred5[j][l], pred6[j][l]]

                            bbox_list = list()
                            for k, prediction in enumerate(pred):
                                bbox_list += bbox.translate_pred_to_bbox(prediction, (args.img_w, args.img_h),
                                                                         args.num_cl,
                                                                             net.pred_infos[k])
                            bbox_list.sort(reverse=True, key=lambda student: student.conf)

                            top_k = 200
                            if len(bbox_list) < top_k:
                                bbox_list = bbox_list[:len(bbox_list)]
                            else:
                                bbox_list = bbox_list[:top_k]

                            bbox_list2 = defaultdict(list)
                            [bbox_list2[bbox_list[k].cl].append(bbox_list[k]) for k in range(len(bbox_list))]

                            bbox_list_nms = list()
                            for k in range(args.num_cl):
                                bbox_list_nms += bbox.nms(bbox_list2[k], 0.45)

                            img = train_batch['image'][j*args.batch_size+l].astype(np.uint8)
                            train_box_img = np.copy(img)

                            [bbox.draw_box(train_box_img, box, inference.label_defs[args.train_set][box.cl].name, inference.label_defs[args.train_set][box.cl].color) for box in bbox_list_nms]
                            train_box_img = cv2.cvtColor(train_box_img, cv2.COLOR_BGR2RGB)
                            train_save_path = os.path.join(args.output_dir, args.ex_dir, 'image/') + 'epoch_{:d}'.format(epoch) + 'img_batch_{:d}'.format(j*args.batch_size+l) + '.png'
                            cv2.imwrite(train_save_path, train_box_img)
                            if args.show_img == True:
                                cv2.imshow('train_box_img', train_box_img)
                                cv2.waitKey(0)

            if epoch % 20 == 0 and epoch is not 0:
                print('save detection_result........................')
                saver = tf.train.Saver()
                snapshot(sess, saver, epoch, args.ex_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='vgg16', type=str,
                        help='Backbone Network vgg16, res50, res101, res152, mobile')
    parser.add_argument('-img_w', '--img_w', type=int, default=300, help='Width of the image.')
    parser.add_argument('-img_h', '--img_h', type=int, default=300, help='Height of the image.')
    parser.add_argument('-num_classes', '--num_cl', type=int, default=0, help='number of classes')
    parser.add_argument('-max_e', '--max_epoch', type=int, default=401, help='Maximum iteration of epoch')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005, help='Weight_decay')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='Learning_rate')
    parser.add_argument('-mom', '--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch_size')
    parser.add_argument('-log_dir', '--log_dir', default='./log_dir', help='log_dir')
    parser.add_argument('-outdir','--output_dir', default='/home/gs/SSD-Tensorflow/output', help='output_dir')
    parser.add_argument('-experimentdir', '--ex_dir', default='model_dump', help='experiment_dir')
    parser.add_argument('-nmsthres','--nms_threshold', type=float, default=0.5, help='nms_threshold')
    parser.add_argument('-checkpoint', '--checkpoint', type=bool, default=0, help='1: load_checkpoint, 0: not')
    parser.add_argument('-num_gpus','--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('-data_set','--train_set', default='VOC',type=str, help='training_set : VOC, KITTI')
    parser.add_argument('-show_img', '--show_img', default=False, type=bool, help='Show the train_box_image')
    parser.add_argument('-save_img', '--save_img', default=True, type=bool, help='Save the train_box_image')
    parser.add_argument('-backbonedir', '--backbone_dir', default='/home/gs/SSD-Tensorflow/vgg16.npy', help='Backbone dir')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train(args)

# batch32, 2*gpu : 75s per epoch
# batch32, 1*gpu : 95s per epoch
