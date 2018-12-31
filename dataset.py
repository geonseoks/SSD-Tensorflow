# ----------------------------------------------------------------
# Tensorflow SSD
#
# Written by Geonseok Seo, based on code from SSD by Lukasz Janyst
# ----------------------------------------------------------------
import os
import tensorflow as tf
import pickle

class Dataset():
    def __init__(self, data_dir, batch_size, mode, num_gpus=1):
        self.mode = mode
        if self.mode == 'train':
            with open(os.path.join(data_dir, 'train-samples.pkl'), 'rb') as fp:
                self.train_samples = pickle.load(fp)

            with open(os.path.join(data_dir, 'train-details.pkl'), 'rb') as fp:
                data = pickle.load(fp)

            self.train_tfs = data['train_transforms']
            self.num_train = len(self.train_samples)
        else:
            with open(os.path.join(data_dir, 'valid-samples.pkl'), 'rb') as fp:
                self.valid_samples = pickle.load(fp)

            with open(os.path.join(data_dir, 'valid-details.pkl'), 'rb') as fp:
                data = pickle.load(fp)

            self.valid_tfs = data['valid_transforms']
            self.valid_train = len(self.valid_samples)

        self.preset = data['preset']
        self.num_classes = data['num_classes']
        self.label_to_idx = data['label2idx']
        self.idx_to_label = data['idx2label']

        if self.mode == 'train':
            self.dataset = tf.data.Dataset.from_tensor_slices(self.train_samples).shuffle(10000)
            self.dataset = self.dataset.map(lambda fname: tuple((tf.py_func(self.run_transforms, [fname], [tf.float32, tf.float32, tf.string]))),num_parallel_calls=6).batch(batch_size*num_gpus, )
            self.dataset = self.dataset.prefetch(buffer_size=1000)
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices(self.valid_samples).shuffle(10000)
            # fname : input, self.run_transforms : tensorflow_op, input shape's type : [float,float,string]
            # map 함수는 함수와 리스트를 인자로 받습니다. 그리고, 리스트로부터 원소를 하나씩 꺼내서 함수를 적용시킨 다음, 그 결과를 새로운 리스트에 담아준답니다.
            self.dataset = self.dataset.map(lambda fname:tuple((tf.py_func(
                self.run_transforms, [fname], [tf.float32, tf.string, tf.int64, tf.int64, tf.double]))),
                num_parallel_calls=6).batch(batch_size*num_gpus, ) # fname을 self.run_transform 이란 함수에 넣음, return type: [tf.float32, tf.float32, tf.string]

    def generator(self):
        if self.mode == 'train':
            for sample in self.train_samples:
                yield None, None, sample
        else:
            for sample in self.valid_samples:
                yield None, None, sample

    def run_transforms(self, fname):
        if not isinstance(fname, str):
            fname = fname.decode('utf-8')
        args = [fname]
        if self.mode == 'train':
            for t in self.train_tfs:
                args = t(*args)
        else:
            for t in self.valid_tfs:
                args = t(*args)
        return args

    def get_iterator(self):
        return self.dataset.make_initializable_iterator()