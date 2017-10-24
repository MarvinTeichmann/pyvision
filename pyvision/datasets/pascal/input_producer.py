"""
Load Kitti Segmentation Input
-------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import logging
import os
import sys
import random
from random import shuffle

import numpy as np

import scipy as scp
import scipy.misc

import socket

import threading


if __name__ == '__main__':
    from cityscape_utils import label_converter as clc
else:
    from cityscape_utils import label_converter as clc

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


NUM_SAMPLES = 4


def default_conf():
    conf = {
        'data': {
            'train_file': 'train2.lst',
            'val_file': 'val.lst',
        },
        'jitter': {
            'fix_shape': False,
            'reseize_image': False,
            'random_resize': False,
            'augment_crop': False,
            'crop_patch': False
        },
        'batch_size': 1


    }

    return conf


class InputProducer():
    """docstring for InputProducer"""

    def __init__(self, conf, data_dir, phase='train'):

        self.phase = phase
        self.data_dir = data_dir

        self.conf = conf

        self.names = ['images', 'labels', 'masks', 'tags']

        assert(phase == 'train' or phase == 'val')

        self.iterator = self._load_voc_data()

    def __iter__(self):
        return self

    def __next__(self):
        inputs = next(self.iterator)
        return dict(zip(self.names, inputs))

    def inputs(self):
        inputs = next(self.iterator)

        return dict(zip(self.names, inputs))

    def _load_voc_data(self):
        """Return a data generator that outputs image samples.

        @ Returns
        image: integer array of shape [width, height, 3].
        Representing RGB value of each pixel.
        gt_image: boolean array of shape [width, height, num_classes].
        Set `gt_image[i,j,k] == 1` if and only if pixel i,j
        is assigned class k. `gt_image[i,j,k] == 0` otherwise.

        [Alternativly make gt_image[i,j,*] a valid propability
        distribution.]
        """
        phase = self.phase
        if phase == 'train':
            data_file = self.conf['data']["train_file"]
        elif phase == 'val':
            data_file = self.conf['data']["val_file"]
        else:
            assert False, "Unknown Phase %s" % phase

        data = self._parse_segmentation_txt(data_file, phase)

        for image, gt_image, load_dict in data:

            if phase == 'val':
                str_dict = np.array(load_dict.__str__())
                labels, masks = self._get_labels_masks(gt_image)

                yield image, labels, masks, str_dict

            elif phase == 'train':

                for i in range(1):

                    '''
                    labels, masks = self._get_labels_masks(gt_image)

                    lm = np.concatenate([12 * labels, 255 * masks,
                                         255 * masks], axis=2)

                    # image_np = image_np[0]
                    scp.misc.imshow(np.concatenate(
                                    [image, lm],
                                    axis=1))
                    '''

                    im, gt_image, load_dict = self.jitter_input(
                        image, gt_image, load_dict)

                    str_dict = np.array(load_dict.__str__())
                    labels, masks = self._get_labels_masks(gt_image)

                    yield im, labels, masks, str_dict

    def jitter_input(self, image, gt_image, load_dict):

        jitter = self.conf['jitter']
        if random.random() > 0.5:
            image = np.fliplr(image)
            gt_image = np.fliplr(gt_image)

        if jitter['reseize_image']:
            image_height = jitter['image_height']
            image_width = jitter['image_width']
            image, gt_image = resize_label_image(
                image, gt_image, image_height, image_width)

        if jitter['random_resize'] and jitter['res_chance'] > random.random():
            lower_size = jitter['lower_size']
            upper_size = jitter['upper_size']
            sig = jitter['sig']
            image, gt_image = random_resize(
                image, gt_image, lower_size, upper_size, sig)

        if jitter['augment_crop'] and jitter['crop_chance'] > random.random():
            image, gt_image = random_crop_soft(image,
                                               gt_image,
                                               jitter['max_crop'])

        if jitter['crop_patch']:
            assert(False)
            patch_height = jitter['patch_height']
            patch_width = jitter['patch_width']
            image, gt_image = self.random_crop(
                image, gt_image, patch_height, patch_width)

        assert(image.shape[:-1] == gt_image.shape)
        return image, gt_image, load_dict

    def _get_labels_masks(self, gt_image):
        """
        Split gt_image into label / mask.

        Parameters
        ----------
        gt_image : numpy array of integer
            Contains numbers encoding labels and 'ignore' area

        Returns
        -------
        labels : numpy array of integer
            Contains numbers 0 to 20, each corresponding to a class
        masks: numpy array of bool
            true, if the pixel is not ignored
        """
        for id in np.unique(gt_image):
            # Check whether gt_image contains valid ids
            valid = id == 255 or id % 10 == 0
            assert(valid)

        masks = gt_image != 255
        labels = masks * gt_image // 10

        masks = masks.reshape(masks.shape + tuple([1]))
        labels = labels.reshape(labels.shape + tuple([1]))

        return labels, masks

    def _parse_segmentation_txt(self, data_file, phase):
        """Take the data_file and hypes and create a generator.

        The generator outputs the image and the gt_image.
        """
        data_base_path = os.path.dirname(__file__)
        data_file = os.path.join(data_base_path, data_file)
        base_path = os.path.realpath(os.path.join(self.data_dir))
        files = [line.rstrip() for line in open(data_file)]

        for epoche in itertools.count():
            if phase == 'train':
                shuffle(files)

            for file in files:
                image_file_raw, gt_image_raw = file.split(" ")
                image_file = os.path.join(base_path, image_file_raw)
                assert os.path.exists(image_file), \
                    "File does not exist: %s" % image_file
                gt_image_file = os.path.join(base_path, gt_image_raw)
                assert os.path.exists(gt_image_file), \
                    "File does not exist: %s" % gt_image_file
                image = scipy.misc.imread(image_file)
                # Please update Scipy, if mode='RGB' is not avaible
                if phase == 'train':
                    gt_image = scp.misc.imread(gt_image_file)
                else:
                    gt_image = scp.misc.imread(gt_image_file)

                load_dict = {'image_file': image_file_raw,
                             'gt_image_file': gt_image_raw,
                             'epoch_end': False}

                yield image, gt_image, load_dict

            if phase == 'val' and False:
                im_shape = image.shape
                gt_shape = gt_image.shape

                load_dict = {'image_file': '',
                             'gt_image_file': '',
                             'epoch_end': True}

                image = np.zeros(im_shape, np.uint8)
                gt = np.zeros(gt_shape, np.uint8)

                yield image, gt, load_dict

    def random_crop(self, image, gt_image, height, width):
        old_width = image.shape[1]
        old_hght = image.shape[0]
        assert(old_width >= width), "image_width: {} crop_width: {}".format(
            old_width, width)
        assert(old_hght >= height), "image_height: {} crop_height: {}".format(
            old_hght, height)

        if self.conf['jitter']['augment_crop']:
            width_factor = width / (old_width + width)
            height_factor = height / (old_hght + height)

            max_y = max(old_width - width, 0)
            if random.random < width_factor:
                if random.random < 0.5:
                    offset_y = 0
                else:
                    offset_y = max_y
            else:
                offset_y = random.randint(0, max_y)

            max_x = max(old_hght - height, 0)
            if random.random < height_factor:
                if random.random < 0.5:
                    offset_x = 0
                else:
                    offset_x = max_x
            else:
                offset_x = random.randint(0, max_x)
        else:
            max_y = max(old_width - width, 0)
            max_x = max(old_hght - height, 0)
            offset_y = random.randint(0, max_y)
            offset_x = random.randint(0, max_x)

        image = image[offset_x:offset_x + height, offset_y:offset_y + width]
        gt_image = gt_image[offset_x:offset_x + height,
                            offset_y:offset_y + width]

        assert(image.shape[0] == height)
        assert(image.shape[1] == width)

        return image, gt_image


def random_crop_soft(image, gt_image, max_crop):
    offset_x = random.randint(1, max_crop)
    offset_y = random.randint(1, max_crop)

    if random.random() > 0.5:
        image = image[offset_x:, offset_y:]
        gt_image = gt_image[offset_x:, offset_y:]
    else:
        image = image[:-offset_x, :-offset_y]
        gt_image = gt_image[:-offset_x, :-offset_y]

    return image, gt_image


def resize_label_image_with_pad(image, label, image_height, image_width):
    shape = image.shape
    assert(image_height >= shape[0])
    assert(image_width >= shape[1])

    pad_height = image_height - shape[0]
    pad_width = image_width - shape[1]
    off_x = random.randint(0, pad_height)
    off_y = random.randint(0, pad_width)

    new_image = np.zeros([image_height, image_width, 3])
    new_image[off_x:off_x + shape[0], off_y:off_y + shape[1]] = image

    new_label = np.zeros([image_height, image_width, 2])
    new_label[off_x:off_x + shape[0], off_y:off_y + shape[1]] = label

    return new_image, new_label


def resize_label_image(image, gt_image,
                       image_height, image_width):

    image = scipy.misc.imresize(image, size=(image_height, image_width),
                                interp='cubic')

    gt_image = scipy.misc.imresize(gt_image,
                                   size=(image_height, image_width),
                                   interp='nearest')
    # https://github.com/scipy/scipy/issues/4458#issuecomment-269067103
    assert(False)

    return image, gt_image


def random_resize(image, gt_image, lower_size, upper_size, sig):
    factor = random.normalvariate(1.1, sig)
    if factor < lower_size:
        factor = lower_size
    if factor > upper_size:
        factor = upper_size
    image = scipy.misc.imresize(image, factor)
    gt_image2 = scipy.misc.imresize(gt_image, factor, interp='nearest')

    return image, gt_image2


def crop_to_size(hypes, image, gt_image):
    new_width = image.shape[1]
    new_height = image.shape[0]
    width = hypes['arch']['image_width']
    height = hypes['arch']['image_height']
    if new_width > width:
        max_x = max(new_height - height, 0)
        max_y = new_width - width
        off_x = random.randint(0, max_x)
        off_y = random.randint(0, max_y)
        image = image[off_x:off_x + height, off_y:off_y + width]
        gt_image = gt_image[off_x:off_x + height, off_y:off_y + width]

    return image, gt_image
