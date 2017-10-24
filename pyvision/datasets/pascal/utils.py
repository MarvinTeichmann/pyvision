"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import scipy.misc

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def pascal_classes():
    classes = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
               'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8,
               'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12,
               'horse': 13, 'motorbike': 14, 'person': 15, 'potted-plant': 16,
               'sheep': 17, 'sofa': 18, 'train': 19, 'tv/monitor': 20}

    return classes


def pascal_palette():
    palette = {(0, 0, 0): 0,
               (128, 0, 0): 1,
               (0, 128, 0): 2,
               (128, 128, 0): 3,
               (0, 0, 128): 4,
               (128, 0, 128): 5,
               (0, 128, 128): 6,
               (128, 128, 128): 7,
               (64, 0, 0): 8,
               (192, 0, 0): 9,
               (64, 128, 0): 10,
               (192, 128, 0): 11,
               (64, 0, 128): 12,
               (192, 0, 128): 13,
               (64, 128, 128): 14,
               (192, 128, 128): 15,
               (0, 64, 0): 16,
               (128, 64, 0): 17,
               (0, 192, 0): 18,
               (128, 192, 0): 19,
               (0, 64, 128): 20}

    return palette


voc_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat',
             'chair', 'cow', 'diningtable', 'dog',
             'horse', 'motorbike', 'person', 'potted-plant',
             'sheep', 'sofa', 'train', 'tv/monitor']

color_list = [(0, 0, 0),
              (128, 0, 0),
              (0, 128, 0),
              (128, 128, 0),
              (0, 0, 128),
              (128, 0, 128),
              (0, 128, 128),
              (128, 128, 128),
              (64, 0, 0),
              (192, 0, 0),
              (64, 128, 0),
              (192, 128, 0),
              (64, 0, 128),
              (192, 0, 128),
              (64, 128, 128),
              (192, 128, 128),
              (0, 64, 0),
              (128, 64, 0),
              (0, 192, 0),
              (128, 192, 0),
              (0, 64, 128)]


NUM_VOC_TRAIN_CLASSES = 21

save_folder = 'SegmentationIds'


def convert_color_to_segid(gt_image):

    shape = gt_image.shape
    gt_reshaped = np.zeros([shape[0], shape[1]], dtype=np.int32)
    mask = np.zeros([shape[0], shape[1]], dtype=np.int32)

    palette = pascal_palette()

    for color, train_id in palette.items():
        gt_label = np.all(gt_image == color, axis=2)
        mask = mask + gt_label
        gt_reshaped = gt_reshaped + 10 * train_id * gt_label

    assert(np.max(mask) == 1)
    np.unique(gt_reshaped)
    assert(np.max(gt_reshaped) <= 200)

    gt_reshaped = gt_reshaped + 255 * (1 - mask)

    return gt_reshaped


def convert_pascal_voc():
    val_file = 'val.lst'
    train_file = 'train.lst'

    data_dir = os.environ['TV_DIR_DATA']

    val_txts = [line.strip().split() for line in open(val_file)]
    train_txts = [line.strip().split() for line in open(train_file)]

    all_data = val_txts + train_txts

    for image_file, gt_file in all_data:
        real_gt_file = os.path.join(data_dir, gt_file)
        gt_image = scp.misc.imread(real_gt_file)

        gt_id_image = convert_color_to_segid(gt_image)

        img_name = os.path.basename(gt_file)

        logging.info('Converting: {}'.format(img_name))
        new_gt_file = os.path.join(data_dir, 'VOC2012', save_folder, img_name)

        scipy.misc.toimage(gt_id_image, cmin=0, cmax=255).save(
            new_gt_file)


def convert_sbd_voc():
    import scipy.io

    datadir = '/data/cvfs/mttt2/DATA/sbd/dataset'
    files = 'all.txt'

    txt_file = os.path.join(datadir, files)
    img_dir = os.path.join(datadir, 'cls')
    new_img_dir = os.path.join(datadir, 'clsIMG')

    if not os.path.exists(new_img_dir):
        os.mkdir(new_img_dir)

    val_txts = [line.strip() for line in open(txt_file)]

    for image_file in val_txts:
        gt_file = os.path.join(img_dir, image_file)

        gt_mat = scp.io.loadmat(gt_file)
        gt_image = gt_mat['GTcls'][0][0][1]
        gt_image = 10 * gt_image

        logging.info('Converting: {}'.format(image_file))

        new_name = image_file.split('.')[0] + '.png'
        new_gt_file = os.path.join(new_img_dir, new_name)

        scipy.misc.toimage(gt_image, cmin=0, cmax=255).save(
            new_gt_file)

if __name__ == '__main__': # NOQA
    logging.info("Hello World.")
    convert_sbd_voc()
