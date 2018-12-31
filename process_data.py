# ----------------------------------------------------------------
# Tensorflow SSD
#
# Written by Geonseok Seo, based on code from SSD by Lukasz Janyst
# ----------------------------------------------------------------
from tqdm import tqdm
import pickle
import argparse
import lxml.etree
from ssdutils import get_preset_by_name
from transforms import *
from collections import namedtuple
Anchor = namedtuple('Anchor', ['center', 'size'])
def build_sampler(overlap, trials):
    return SamplerTransform(sample=True, min_scale=0.3, max_scale=1.0,
                            min_aspect_ratio=0.5, max_aspect_ratio=2.0,
                            min_jaccard_overlap=overlap, max_trials=trials)
def build_train_transform(preset, num_classes, sampler_trials, expand_prob, label2idx, dataset):
    """
    Data augmentation transforms
    """
    tf_loader = ImageLoaderTransform(label2idx=label2idx, dataset=dataset) # return type: img, labels, sample

    tf_brightness = BrightnessTransform(lower=0.5, upper=1.5)
    tf_rnd_brightness = RandomTransform(prob=0.5, transform=tf_brightness)

    tf_contrast = ContrastTransform(lower=0.5, upper=1.5)
    tf_rnd_contrast = RandomTransform(prob=0.5, transform=tf_contrast)

    tf_hue = HueTransform(lower=-0.5, upper=0.5)
    tf_rnd_hue = RandomTransform(prob=0.5, transform=tf_hue)

    tf_saturation = SaturationTransform(lower=0.5, upper=1.5)
    tf_rnd_saturation = RandomTransform(prob=0.5, transform=tf_saturation)

    tf_reorder_channels = ReorderChannelsTransform()
    tf_rnd_reorder_channels = RandomTransform(prob=0.5, transform=tf_reorder_channels)

    tf_distort_lst = [
                 tf_rnd_contrast,
                 tf_rnd_saturation,
                 tf_rnd_hue,
                 tf_rnd_contrast
    ]
    tf_distort_1 = ComposeTransform(transforms=tf_distort_lst[:-1])
    tf_distort_2 = ComposeTransform(transforms=tf_distort_lst[1:])
    tf_distort_comp = [tf_distort_1, tf_distort_2]
    tf_distort = TransformPickerTransform(transforms=tf_distort_comp)

    tf_expand = ExpandTransform(max_ratio=4.0, mean_value=(104, 117, 123))
    tf_rnd_expand = RandomTransform(prob=expand_prob, transform=tf_expand)

    samplers = [
        SamplerTransform(sample=False),
        #build_sampler(0.1, sampler_trials),# return None(no gt) or box(can make gt)
        build_sampler(0.3, sampler_trials),# return None(no gt) or box(can make gt)
        build_sampler(0.5, sampler_trials),# return None(no gt) or box(can make gt)
        build_sampler(0.7, sampler_trials),# return None(no gt) or box(can make gt)
        build_sampler(0.9, sampler_trials),# return None(no gt) or box(can make gt)
        build_sampler(1.0, sampler_trials) # return None(no gt) or box(can make gt)
    ]
    tf_sample_picker = SamplePickerTransform(samplers=samplers)
    # Horizontal flip
    tf_flip = HorizontalFlipTransform()
    tf_rnd_flip = RandomTransform(prob=0.5, transform=tf_flip)

    tf_Label = LabelTransform(preset=preset, num_classes=num_classes) # return type : data, vec, gt.filename

    tf_resize = ResizeTransform(width=preset.image_size.w,
                                height=preset.image_size.h,
                                algorithms=[Image.BILINEAR,
                                            Image.NEAREST,
                                            Image.BICUBIC,
                                            Image.LANCZOS,
                                            Image.ANTIALIAS])

    transform = [tf_loader, tf_rnd_brightness, tf_distort, tf_rnd_reorder_channels, tf_rnd_expand, tf_sample_picker, tf_rnd_flip, tf_resize, tf_Label]
    return transform

def build_valid_transforms(preset, num_classes, label2idx, dataset):
    tf_resize = ResizeTransform(width=preset.image_size.w, height=preset.image_size.h, algorithms=[Image.BILINEAR])
    transforms = [
        ImageLoaderTransform(label2idx=label2idx, dataset=dataset),
        tf_resize,
        LabelTransform_valid(preset=preset, num_classes=num_classes)
    ]
    return transforms

class DATASETSource:
    def __init__(self, data_set):
        if data_set == 'VOC':
            self.label_to_idx = {'dog':0, 'sofa':1, 'chair':2, 'boat':3, 'bicycle':4, 'car':5,
                                 'pottedplant':6, 'horse':7, 'cow':8, 'person':9,'diningtable':10,'bottle':11,
                                 'cat':12,'bird':13,'train':14,'motorbike':15,'tvmonitor':16,'aeroplane':17,'bus':18,'sheep':19}
            self.idx_to_label = {0:'dog', 1:'sofa', 2:'chair', 3:'boat', 4:'bicycle', 5:'car', 6:'pottedplant',
                                 7:'horse', 8:'cow', 9:'person', 10:'diningtable', 11:'bottle', 12:'cat', 13:'bird',
                                 14:'train', 15:'motorbike', 16:'tvmonitor', 17:'aeroplane', 18:'bus', 19:'sheep'}
        elif data_set == 'KITTI':
            self.label_to_idx = {'car':0,'truck':1,'misc':2,'van':3,'pedestrian':4,'cyclist':5,'tram':6,'person_sitting':7}
            self.idx_to_label = {0:'car',1:'truck',2:'misc',3:'van',4:'pedestrian',5:'cyclist',6:'tram',7:'person_sitting'}
        self.num_train = 0
        self.num_valid = 0

    def get_DATASET(self, root, mode, dataset = 'VOC'):
        samples = []

        if dataset == 'VOC':
            annot_root = root + '/Annotations/'
            annot_files = []

            with open(root + '/ImageSets/Main/' + mode + '.txt') as f:
                for line in f:
                    annot_file = annot_root + line.strip() + '.xml'
                    if os.path.exists(annot_file):
                        annot_files.append(annot_file)

            image_root = root + '/JPEGImages/'
            for annot in tqdm(annot_files, desc='# VOC image data loading' + '(' + mode + ')'):
                with open(annot, 'r') as f: # open 000002.xml file
                    doc = lxml.etree.parse(f)
                    sample = image_root + doc.xpath('/annotation/filename')[0].text# filename : ~/JPEGImages/000002.jpg
                    samples.append(sample)

        elif dataset == 'KITTI':
            if mode is not 'test':
                root = os.path.join(root, 'training')
            else:
                root = os.path.join(root, 'testing')

            for fname in tqdm(os.listdir(os.path.join(root, 'image_2')), desc='# KITTI data loading'):
                with open(os.path.join(root, 'image_2', fname),'rb') as fp:
                    with Image.open(fp) as img:
                        img_size = Size(img.size[0], img.size[1])

                with open(os.path.join(root, 'label_2', fname[:-4] + '.txt'), 'r') as fp:
                    objs = [line.split(' ') for line in fp.readlines()]

                boxes = []
                for obj in objs:
                    if obj[0] == 'DontCare':
                        continue
                    xmin = float(obj[4])
                    ymin = float(obj[5])
                    xmax = float(obj[6])
                    ymax = float(obj[7])
                    label = self.label_to_idx[obj[0]]
                    center, size = abs2prop(xmin, xmax, ymin, ymax, img_size)
                    box = Box(obj[0], label, center, size)
                    boxes.append(box)
                if not boxes:
                    continue

                sample = os.path.join(root, 'image_2', fname)
                samples.append(sample)

        return samples

    def load_trainval_data(self, data_dir, data_set):
        """
        Load the training and validation data
        Args:
            data_dir:       the directory where the dataset's file are stored
            valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """
        self.train_samples = self.get_DATASET(data_dir, 'trainval', data_set)
        self.num_train = len(self.train_samples)
        self.num_classes = len(self.label_to_idx)
        if len(self.train_samples) == 0:
            raise RuntimeError('No training samples found ' + data_dir)

    def load_validval_data(self, data_dir, data_set):
        """
        Load the training and validation data
        Args:
            data_dir:       the directory where the dataset's file are stored
            valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """
        self.valid_samples = self.get_DATASET(data_dir, 'test', data_set)
        self.num_valid = len(self.valid_samples)
        self.num_classes = len(self.label_to_idx)
        if len(self.valid_samples) == 0:
            raise RuntimeError('No training samples found ' + data_dir)

def main():
    parser = argparse.ArgumentParser(description='Data prepropcess for SSD')
    parser.add_argument('--data_dir', default='/home/gs/data/VOCdevkit.../VOC2007')# KITTI PATH: '/home/gs/data/KITTI'
    parser.add_argument('--train_save_dir', default=os.path.dirname(__file__))
    parser.add_argument('--preset', default='vgg300', choices=['vgg300', 'vgg512'])
    parser.add_argument('--data_set', default='VOC', choices=['VOC', 'KITTI'])
    parser.add_argument('--sampler_trials', type=int, default=50)
    parser.add_argument('--expand_probability', type=float, default=0.5)
    args = parser.parse_args()

    source = DATASETSource(args.data_set)
    source.load_trainval_data(args.data_dir, args.data_set)

    preset = get_preset_by_name(args.preset)

    with open(os.path.join(args.train_save_dir, 'train-samples.pkl'), 'wb') as fp:
        pickle.dump(source.train_samples, fp)

    # pickle 모듈을 이용하면 원하는 데이터를 자료형의 변경없이 파일로 저장하여 그대로 로드할 수 있다.
    with open(os.path.join(args.train_save_dir, 'train-details.pkl'), 'wb') as fp: #pickle로 데이터를 저장하거나 불러올때는 파일을 바이트형식으로 읽거나 써야한다. (wb, rb)
        data = {
            'preset': preset,
            'num_classes': source.num_classes,
            'idx2label': source.idx_to_label,
            'label2idx': source.label_to_idx,
            'train_transforms': build_train_transform(preset, source.num_classes, 
                                                        args.sampler_trials,
                                                        args.expand_probability,
                                                        source.label_to_idx,
                                                      args.data_set)
        }
        pickle.dump(data, fp)

    source = DATASETSource(args.data_set)
    source.load_validval_data(args.data_dir, args.data_set)

    with open(os.path.join(args.train_save_dir, 'valid-samples.pkl'), 'wb') as fp:
        pickle.dump(source.valid_samples, fp)

    # pickle 모듈을 이용하면 원하는 데이터를 자료형의 변경없이 파일로 저장하여 그대로 로드할 수 있다.
    with open(os.path.join(args.train_save_dir, 'valid-details.pkl'), 'wb') as fp: #pickle로 데이터를 저장하거나 불러올때는 파일을 바이트형식으로 읽거나 써야한다. (wb, rb)
        data = {
            'preset': preset,
            'num_classes': source.num_classes,
            'idx2label': source.idx_to_label,
            'label2idx': source.label_to_idx,
            'valid_transforms': build_valid_transforms(preset,source.num_classes, source.label_to_idx ,
                                                      args.data_set)
        }
        pickle.dump(data, fp)

    return 0

if __name__=='__main__':
    main()


