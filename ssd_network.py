# ----------------------------------------------------------------
# Tensorflow SSD
#
# Written by Geonseok Seo, based on code from SSD by Lukasz Janyst
# ----------------------------------------------------------------

import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import namedtuple
import numpy as np
import inspect
import os
PredInfo = namedtuple('PredInfo', ['size', 'dbox_scale', 'asp_ratios', 'extra_scale'])

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

class SSD(object):
    def __init__(self, args, backbone_dir):
        self.args = args
        self.backbone = Vgg16(backbone_dir)
        self.input_nodes = dict()
        self.output_nodes = dict()
        self.loss_nodes = dict()
        if args.img_h == 300:
            self.pred_hs = [38, 19, 10, 5, 3, 1]
        elif args.img_h == 500:
            self.pred_hs = [38, 19, 10, 5, 3, 1]
        else:
            assert args.img_h in [300, 500]

        if args.img_w == 300:
            self.pred_ws = [38, 19, 10, 5, 3, 1]
        elif args.img_w == 500:
            self.pred_ws = [38, 19, 10, 5, 3, 1]
        else:
            assert args.img_w in [300, 500]

        self.pred_dboxs = [4, 6, 6, 6, 4, 4]

        sum = 0
        for i in range(6):
            sum = sum + self.pred_hs[i] * self.pred_ws[i] * self.pred_dboxs[i]
        self.num_dbox = sum

        self.pred_infos = [
            PredInfo(size=(self.pred_hs[0], self.pred_ws[0]), dbox_scale=0.1,
                     asp_ratios=(1, 2.0, 0.5, 1), extra_scale=0.2),
            PredInfo(size=(self.pred_hs[1], self.pred_ws[1]), dbox_scale=0.2,
                     asp_ratios=(1, 2.0, 3.0, 0.5, 1.0 / 3.0, 1), extra_scale=0.375),
            PredInfo(size=(self.pred_hs[2], self.pred_ws[2]), dbox_scale=0.375,
                     asp_ratios=(1, 2.0, 3.0, 0.5, 1.0 / 3.0, 1), extra_scale=0.55),
            PredInfo(size=(self.pred_hs[3], self.pred_ws[3]), dbox_scale=0.55,
                     asp_ratios=(1, 2.0, 3.0, 0.5, 1.0 / 3.0, 1), extra_scale=0.725),
            PredInfo(size=(self.pred_hs[4], self.pred_ws[4]), dbox_scale=0.725,
                     asp_ratios=(1, 2.0, 0.5, 1), extra_scale=0.9),
            PredInfo(size=(self.pred_hs[5], self.pred_ws[5]), dbox_scale=0.9,
                     asp_ratios=(1, 2.0, 0.5, 1), extra_scale=1.075)]
        self.image = tf.placeholder(tf.float32, shape=[None, self.args.img_h, self.args.img_w, 3], name='image')
        self.gt_cl = tf.placeholder(tf.float32, shape=[None, self.num_dbox, self.args.num_cl + 1],
                                    name='gt_cl')
        self.gt_loc = tf.placeholder(tf.float32, shape=[None, self.num_dbox, 4], name='gt_loc')

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            init = tf.constant_initializer(value=self.backbone.data_dict[name][0])
            filt = tf.get_variable(name="filter", shape=init.value.shape, initializer=init)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            init2 = tf.constant_initializer(value=self.backbone.data_dict[name][1])
            conv_biases = tf.get_variable(name="biases", shape=init2.value.shape, initializer=init2)

            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def create_network(self, batch_start, batch_end):

        with tf.variable_scope('network'):

            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.image[batch_start:batch_end])
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            self.bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
            # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

            self.conv1_1 = self.conv_layer(self.bgr, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")

            #initializer = tf.initializers.random_normal(0.0, 0.001)
            initializer = tf.contrib.layers.xavier_initializer()
            pool5 = slim.max_pool2d(self.conv5_3, [3, 3], stride=1, scope='conv5_maxpool', padding='SAME')

            orig_w, orig_b = self.backbone.data_dict['fc6'][0], self.backbone.data_dict['fc6'][1]
            orig_w = orig_w.reshape(7, 7, 512, 4096)
            mod_w = np.zeros((3, 3, 512, 1024))
            mod_b = np.zeros(1024)
            for i in range(1024):
                mod_b[i] = orig_b[4 * i]
                for h in range(3):
                    for w in range(3):
                        mod_w[h, w, :, i] = orig_w[3 * h, 3 * w, :, 4 * i]

            def array2tensor(x, name):
                init = tf.constant_initializer(value=x, dtype=tf.float32)
                tensor = tf.get_variable(name=name, initializer=init, shape=x.shape)
                return tensor

            w = array2tensor(mod_w, 'conv6_filter')
            b = array2tensor(mod_b, 'conv6_biases')
            x = tf.nn.atrous_conv2d(pool5, w, rate=6, padding='SAME')
            x = tf.nn.bias_add(x, b)
            conv6 = tf.nn.relu(x)

            orig_w, orig_b = self.backbone.data_dict['fc7'][0], self.backbone.data_dict['fc7'][1]
            orig_w = orig_w.reshape(1, 1, 4096, 4096)
            mod_w = np.zeros((1, 1, 1024, 1024))
            mod_b = np.zeros(1024)
            for i in range(1024):
                mod_b[i] = orig_b[4 * i]
                for j in range(1024):
                    mod_w[:, :, j, i] = orig_w[:, :, 4 * j, 4 * i]

            w = array2tensor(mod_w, 'conv7_filter')
            b = array2tensor(mod_b, 'conv7_biases')
            x = tf.nn.conv2d(conv6, w, strides=[1, 1, 1, 1],
                             padding='SAME')
            x = tf.nn.bias_add(x, b)
            conv7 = tf.nn.relu(x)

            conv8_1 = slim.conv2d(conv7, 256, [1, 1], weights_initializer=initializer, scope='conv8')
            conv8_2 = slim.conv2d(conv8_1, 512, [3, 3], stride=2, weights_initializer=initializer, scope='conv8_2', padding='SAME')
            conv9_1 = slim.conv2d(conv8_2, 128, [1, 1], weights_initializer=initializer, scope='conv9_1')
            conv9_2 = slim.conv2d(conv9_1, 256, [3, 3], weights_initializer=initializer, stride=2, scope='conv9_2', padding='SAME')
            conv10_1 = slim.conv2d(conv9_2, 128, [1, 1], weights_initializer=initializer, scope='conv10_1')
            conv10_2 = slim.conv2d(conv10_1, 256, [3, 3], weights_initializer=initializer, scope='conv10_2', padding='VALID')
            conv11_1 = slim.conv2d(conv10_2, 128, [1, 1], weights_initializer=initializer, scope='conv11_1')
            conv11_2 = slim.conv2d(conv11_1, 256, [3, 3], weights_initializer=initializer, scope='conv11_2', padding='VALID')

            with tf.variable_scope('l2_norm_conv4_3'):

                scale = array2tensor(20 * np.ones(512), 'scale')
                x = scale * tf.nn.l2_normalize(self.conv4_3, axis=-1)

            pred1 = slim.conv2d(x, 4 * (self.args.num_cl + 5), [3, 3], weights_initializer=initializer, scope='pred1')
            pred2 = slim.conv2d(conv7, 6 * (self.args.num_cl + 5), [3, 3], weights_initializer=initializer, scope='pred2')
            pred3 = slim.conv2d(conv8_2, 6 * (self.args.num_cl + 5), [3, 3], weights_initializer=initializer, scope='pred3')
            pred4 = slim.conv2d(conv9_2, 6 * (self.args.num_cl + 5), [3, 3], weights_initializer=initializer, scope='pred4')
            pred5 = slim.conv2d(conv10_2, 4 * (self.args.num_cl + 5), [3, 3], weights_initializer=initializer, scope='pred5')
            pred6 = slim.conv2d(conv11_2, 4 * (self.args.num_cl + 5), [3, 3], weights_initializer=initializer, scope='pred6')

            self.output_nodes['pred1'] = pred1
            self.output_nodes['pred2'] = pred2
            self.output_nodes['pred3'] = pred3
            self.output_nodes['pred4'] = pred4
            self.output_nodes['pred5'] = pred5
            self.output_nodes['pred6'] = pred6

            tf.add_to_collection("bbox_pred1", pred1)
            tf.add_to_collection("bbox_pred2", pred2)
            tf.add_to_collection("bbox_pred3", pred3)
            tf.add_to_collection("bbox_pred4", pred4)
            tf.add_to_collection("bbox_pred5", pred5)
            tf.add_to_collection("bbox_pred6", pred6)

    def create_basic_loss(self, batch_start, batch_end):
        with tf.variable_scope('prediction'):
            reshape_output_nodes = list()
            reshape_tensorss = list()
            for i, output_node in enumerate(self.output_nodes):
                reshape_output_nodes.append(tf.split(self.output_nodes[output_node],self.pred_dboxs[i],-1))
                reshape_tensors = list()
                for reshape_tensor in reshape_output_nodes[i]:
                    reshape_tensors.append(tf.reshape(reshape_tensor,[-1,self.pred_hs[i]*self.pred_ws[i],self.args.num_cl + 5]))
                reshape_tensorss.append(tf.concat(reshape_tensors, 1))
            total_pred = tf.concat(reshape_tensorss, 1) # ((batch*38*38*4*class, 13) , (batch*19*19*6*class, 13) ... )

            pred_cl = total_pred[:, :, :self.args.num_cl + 1] # ((batch*38*38*4*class, 9) , (batch*19*19*6*class, 9) ... )
            pred_loc = total_pred[:, :, self.args.num_cl + 1:] # ((batch*38*38*4*class,10~14) , (batch*19*19*6*class, 10~14) ... )

        with tf.variable_scope('ground_truth'):
            gt_cl = self.gt_cl[batch_start:batch_end]
            gt_loc = self.gt_loc[batch_start:batch_end]

        # TODO: check variations of cross-entropy function
        def smooth_l1_loss(x):
            square_loss = 0.5 * x ** 2
            absolute_loss = tf.abs(x)
            return tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss - 0.5)

        #-----------------------------------------------------------------------
        # Compute match counters
        #-----------------------------------------------------------------------
        with tf.variable_scope('match_counters'):
            # Number of anchors per sample
            # Shape: (batch_size)
            total_num = tf.ones([self.args.batch_size], dtype=tf.int64) * tf.to_int64(self.num_dbox) # (8732, 8732, ...) 32개

            # Number of negative (not-matched) anchors per sample, computed
            # by counting boxes of the background class in each sample.
            # Shape: (batch_size)
            negatives_num = tf.count_nonzero(gt_cl[:,:,-1], axis=1) # 각 batch에서 모든 anchor(8732개) 에서 background가 0이아닌 갯수 = num of negatives
                                                                    # ex) batch3일때 , negatvies_num의 shape = [3000개,4000개,2000개]

            # Number of positive (matched) anchors per sample
            # Shape: (batch_size)
            positives_num = total_num - negatives_num

            #self.output_nodes['positives_num'] = positives_num
            #self.output_nodes['negatives_num'] = negatives_num

            # Number of positives per sample that is division-safe
            # Shape: (batch_size)
            positives_num_safe = tf.where(tf.equal(positives_num, 0),
                                          tf.ones([self.args.batch_size]) * 1e-6,          #나누기 0 방지 #[0.00001, 0.00001, ...] 32개
                                          tf.to_float(positives_num))                      # [10,400,30, 0...] batch당 positive개수 (float형)

        #-----------------------------------------------------------------------
        # Compute masks
        #-----------------------------------------------------------------------
        with tf.variable_scope('match_masks'):
            # Boolean tensor determining whether an anchor is a positive
            # Shape: (batch_size, num_anchors)
            positives_mask = tf.equal(gt_cl[:,:,-1], 0)                               # 8732개 당 positive면 1 -> i번째 batch,j번째 anchor는 positive

            # Boolean tensor determining whether an anchor is a negative
            # Shape: (batch_size, num_anchors)
            negatives_mask = tf.logical_not(positives_mask)

        #-----------------------------------------------------------------------
        # Compute the confidence loss
        #-----------------------------------------------------------------------
        with tf.variable_scope('confidence_loss'):
            # Cross-entropy tensor - all of the values are non-negative
            # Shape: (batch_size, num_anchors)
            ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_cl, logits=pred_cl)

            #-------------------------------------------------------------------
            # Sum up the loss of all the positive anchors
            #-------------------------------------------------------------------
            # Positives - the loss of negative anchors is zeroed out
                                                                                           # Shape: (batch_size, num_anchors)
            positives = tf.where(positives_mask, ce, tf.zeros_like(ce))                    # positive mask만 ce loss 남겨둠

            #self.output_nodes['positives'] = positives
            # Total loss of positive anchors
                                                                                           # Shape: (batch_size)
            positives_sum = tf.reduce_sum(positives, axis=-1)                              # positive mask sum

            #-------------------------------------------------------------------
            # Figure out what the negative anchors with highest confidence loss
            # are
            #-------------------------------------------------------------------
            # Negatives - the loss of positive anchors is zeroed out
                                                                                           # Shape: (batch_size, num_anchors)
            negatives = tf.where(negatives_mask, ce, tf.zeros_like(ce))                    # negative mask만 ce loss 남겨둠

            # Top negatives - sorted confience loss with the highest one first
                                                                                           # Shape: (batch_size, num_anchors)
            negatives_top = tf.nn.top_k(negatives, self.num_dbox)[0]                       # ce loss 높은순으로 .. [0] : value, [1] : index
            #self.output_nodes['negatives_max_mask'] = negatives_top
            #self.output_nodes['negatives_max_mask2'] = tf.nn.top_k(negatives, self.num_dbox)

            #-------------------------------------------------------------------
            # Fugure out what the number of negatives we want to keep is
            #-------------------------------------------------------------------
            # Maximum number of negatives to keep per sample - we keep at most
            # 3 times as many as we have positive anchors in the sample
            # Shape: (batch_size)
            negatives_num_max = tf.minimum(negatives_num, 3 * positives_num)               # [30, 1200, 90, ...]

            #-------------------------------------------------------------------
            # Mask out superfluous negatives and compute the sum of the loss
            #-------------------------------------------------------------------
            # Transposed vector of maximum negatives per sample
            # Shape (batch_size, 1)                                                        # [30, 1200, 90, ...]
            negatives_num_max_t = tf.expand_dims(negatives_num_max, 1) # 크기 1인 차원을 텐서의 구조(shape)에 삽입함

            # Range tensor: [0, 1, 2, ..., num_anchors-1]
            # Shape: (num_anchors)
            rng = tf.range(0, self.num_dbox, 1)

            # Row range, the same as above, but int64 and a row of a matrix
            # Shape: (1, num_anchors)
            range_row = tf.to_int64(tf.expand_dims(rng, 0))

            # Mask of maximum negatives - first `negative_num_max` elements
            # in corresponding row are `True`, the rest is false
            # Shape: (batch_size, num_anchors)
            negatives_max_mask = tf.less(range_row, negatives_num_max_t)

            # Max negatives - all the positives and superfluous negatives are
            # zeroed out.
            # Shape: (batch_size, num_anchors)
            #self.output_nodes['negatives_max_mask'] = negatives_max_mask
            negatives_max = tf.where(negatives_max_mask, negatives_top,
                                     tf.zeros_like(negatives_top))

            # Sum of max negatives for each sample
            # Shape: (batch_size)
            negatives_max_sum = tf.reduce_sum(negatives_max, axis=-1)

            #-------------------------------------------------------------------
            # Compute the confidence loss for each element
            #-------------------------------------------------------------------
            # Total confidence loss for each sample
            # Shape: (batch_size)
            confidence_loss = tf.add(positives_sum, negatives_max_sum)

            # Total confidence loss normalized by the number of positives
            # per sample
            # Shape: (batch_size)
            confidence_loss = tf.where(tf.equal(positives_num, 0),
                                       tf.zeros([self.args.batch_size]),
                                       tf.div(confidence_loss,
                                              positives_num_safe))

            # Mean confidence loss for the batch
            # Shape: scalar
            confidence_loss = tf.reduce_mean(confidence_loss,
                                                  name='confidence_loss')

        #-----------------------------------------------------------------------
        # Compute the localization loss
        #-----------------------------------------------------------------------
        with tf.variable_scope('localization_loss'):
            # Element-wise difference between the predicted localization loss
            # and the ground truth
            # Shape: (batch_size, num_anchors, 4)
            loc_diff = tf.subtract(pred_loc, gt_loc)

            # Smooth L1 loss
            # Shape: (batch_size, num_anchors, 4)
            loc_loss = smooth_l1_loss(loc_diff)

            # Sum of localization losses for each anchor
            # Shape: (batch_size, num_anchors)
            loc_loss_sum = tf.reduce_sum(loc_loss, axis=-1)

            # Positive locs - the loss of negative anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            positive_locs = tf.where(positives_mask, loc_loss_sum,
                                     tf.zeros_like(loc_loss_sum))

            # Total loss of positive anchors
            # Shape: (batch_size)
            localization_loss = tf.reduce_sum(positive_locs, axis=-1)

            # Total localization loss normalized by the number of positives
            # per sample
            # Shape: (batch_size)
            localization_loss = tf.where(tf.equal(positives_num, 0),
                                         tf.zeros([self.args.batch_size]),
                                         tf.div(localization_loss,
                                                positives_num_safe))

            # Mean localization loss for the batch
            # Shape: scalar
            localization_loss = tf.reduce_mean(localization_loss, name='localization_loss')

        # L2 regularization
        with tf.variable_scope('weight_decay'):
            l2_regul = tf.contrib.layers.l2_regularizer(self.args.weight_decay)
            regul_loss = tf.contrib.layers.apply_regularization(
                l2_regul, weights_list=tf.contrib.framework.get_trainable_variables()[:46]+tf.contrib.framework.get_trainable_variables()[47:])

        #-----------------------------------------------------------------------
        # Compute total loss
        #-----------------------------------------------------------------------
        with tf.variable_scope('total_loss'):
            # Sum of the localization and confidence loss
            # Shape: (batch_size)

            total_loss = tf.add(confidence_loss, localization_loss)
            total_loss = tf.add(total_loss, regul_loss)

        return confidence_loss, localization_loss, regul_loss, total_loss