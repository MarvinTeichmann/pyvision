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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging

from mutils.image import normalize

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class SegmentationVisualizer2(object):
    """docstring for ClassificationVisualizer"""

    def __init__(self, names=None, colours=None,
                 mode='RGB'):
        self.names = names
        self.colours = colours

        self.num_classes = len(self.names)

        self.ignore_idx = -100
        self.mask_color = [125, 125, 125]

        if mode == 'RGB':
            self.chan = 3
        else:
            raise NotImplementedError

    def plot(self, image, label, pred, name=None, idx=0):

        fig, ax = plt.subplots(2, 3, figsize=(10, 6))

        mask = label != self.ignore_idx

        ax[0][0].set_title("Image: {}".format(name))
        ax[0][0].imshow(normalize(image))
        ax[0][0].set_xlabel("Index: {}".format(idx))

        clabel = self.id2colour(label)
        ax[1][0].set_title("Label: {}".format(name))
        ax[1][0].imshow(normalize(clabel))
        ax[1][0].set_xlabel("Index: {}".format(idx))

        cpred = self.id2colour(np.argmax(pred, axis=0), mask)

        ax[0][1].set_title("Pred Hard")
        ax[0][1].imshow(normalize(image))
        ax[0][1].imshow(normalize(cpred), alpha=0.5)
        ax[0][1].set_xlabel("Index: {}".format(idx))

        cpred2 = self.pred2colour(pred, mask)
        ax[1][1].set_title("Pred Soft")
        ax[1][1].imshow(normalize(cpred2))
        ax[1][1].set_xlabel("Index: {}".format(idx))

        diff = 1 - self.compute_diff(label, pred)
        ax[0][2].set_title("Diff overlay")
        ax[0][2].imshow(normalize(image))
        ax[0][2].imshow(diff, vmin=0, vmax=1, alpha=0.5, cmap='bwr')
        ax[0][2].set_xlabel("Index: {}".format(idx))

        ax[1][2].set_title("Diff")
        ax[1][2].imshow(diff, vmin=0, vmax=1, cmap='bwr')
        ax[1][2].set_xlabel("Index: {}".format(idx))

        return fig

    def compute_diff(self, label, pred):
        diff = np.zeros(label.shape)
        for i in range(self.num_classes):
            tp = label == i
            diff[tp] += pred[i][tp]

        diff += (label == self.ignore_idx) * 0.5

        return diff

    def id2colour(self, label, mask=None):
        colour_gt = np.zeros(list(label.shape) + [self.chan])
        label = np.expand_dims(label, -1)
        for i, colour in enumerate(self.colours):
            colour_gt += (label == i) * np.array(colour)

        if mask is None:
            mask = label != self.ignore_idx
        else:
            mask = np.expand_dims(mask, -1)

        colour_gt += (1 - mask) * np.array([125, 125, 125])

        return colour_gt.astype(np.uint8)

    def pred2colour(self, pred, mask=None):

        color_image = np.dot(
            np.transpose(pred, [1, 2, 0]), self.colours)

        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.reshape(mask.shape + tuple([1]))

            color_image = mask * color_image + (1 - mask) * self.mask_color

        return color_image

    def underlay2(self, image, gt_image, labels):
        # TODO
        color_img = self.id2color(gt_image)
        color_labels = self.id2color(labels)

        output = np.concatenate((image, color_img, color_labels), axis=0)

        return output

    def overlay(self, image, gt_image, t=0.6):
        # TODO
        # color_img = self.id2color((gt_image))
        output = t * normalize(gt_image) + \
            (1 - t) * normalize(image)

        return output


class SegmentationVisualizer(object):
    """docstring for label_converter"""
    def __init__(self, color_list=None, name_list=None,
                 mode='RGB'):
        super(SegmentationVisualizer, self).__init__()
        self.color_list = color_list
        self.name_list = name_list

        self.mask_color = [255, 255, 255]

        if mode == 'RGB':
            self.chan = 3

    def id2color(self, id_image, mask=None, ignore_idx=-100):
        """
        Input: Int Array of shape [height, width]
            Containing Integers 0 <= i <= num_classes.
        """

        if mask is None:
            if np.any(id_image != ignore_idx):
                mask = id_image != ignore_idx

        shape = id_image.shape
        gt_out = np.zeros([shape[0], shape[1], self.chan], dtype=np.int32)
        id_image

        for train_id, color in enumerate(self.color_list):
            c_mask = id_image == train_id
            c_mask = c_mask.reshape(c_mask.shape + tuple([1]))
            gt_out = gt_out + color * c_mask

        if mask is not None:
            mask = mask.reshape(mask.shape + tuple([1]))
            bg_color = [0, 0, 0]
            mask2 = np.all(gt_out == bg_color, axis=2)
            mask2 = mask2.reshape(mask2.shape + tuple([1]))
            gt_out = gt_out + mask2 * (self.mask_color * (1 - mask))

        return gt_out

    def pred2color(self, pred_image, mask=None):

        color_image = np.dot(pred_image, self.color_list)

        if mask is not None:

            if len(mask.shape) == 2:
                mask = mask.reshape(mask.shape + tuple([1]))

            color_image = mask * color_image + (1 - mask) * self.mask_color

        return color_image

    def color2id(self, color_gt):
        assert(False)
        shape = color_gt.shape
        gt_reshaped = np.zeros([shape[0], shape[1]], dtype=np.int32)
        mask = np.zeros([shape[0], shape[1]], dtype=np.int32)

        for train_id, color in enumerate(self.color_list):
            gt_label = np.all(color_gt == color, axis=2)
            mask = mask + gt_label
            gt_reshaped = gt_reshaped + 10 * train_id * gt_label

        assert(np.max(mask) == 1)
        np.unique(gt_reshaped)
        assert(np.max(gt_reshaped) <= 200)

        gt_reshaped = gt_reshaped + 255 * (1 - mask)
        return gt_reshaped

    def underlay2(self, image, gt_image, labels):
        # TODO
        color_img = self.id2color(gt_image)
        color_labels = self.id2color(labels)

        output = np.concatenate((image, color_img, color_labels), axis=0)

        return output

    def overlay(self, image, gt_image):
        # TODO
        color_img = self.id2color((gt_image))
        output = 0.4 * color_img[:, :] + 0.6 * image

        return output
