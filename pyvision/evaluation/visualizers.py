"""
The MIT License (MIT)

Copyright (c) 2019 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pyvision as pv2

# matplotlib.use('agg')
# matplotlib.use('TkAgg')

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class BinarySegVisualizer():

    def __init__(self):
        self.ignore_idx = -100

    def coloured_diff(self, label, pred, mask):
        if self.label_type == 'dense':
            true_colour = [0, 0, 255]
            false_colour = [255, 0, 0]

            pred = np.argmax(pred, axis=0)

            diff_img = 1 * (pred == label)
            diff_img = diff_img + (1 - mask)

            diff_img = np.expand_dims(diff_img, axis=-1)

            assert(np.max(diff_img) <= 1)

            return true_colour * diff_img + false_colour * (1 - diff_img)

    def plot_prediction(self, prediction, label, image,
                        trans=0.66, figure=None):

        if figure is None:
            figure = plt.figure()
            figure.tight_layout()

        ignore = self.ignore_idx == label

        label = label.astype(np.float).copy()
        label[ignore] = 0.5

        bwr_map = cm.get_cmap('bwr')
        colour_pred = bwr_map(prediction[1])
        colour_label = bwr_map(label)

        # label_r = label.reshape(label.shape + tuple([1]))

        # colour_label2 = [1, 0, 0] * label_r + [0, 0, 1] * (1 - label_r)
        hard_pred = prediction[0] < 0.7
        colour_hard = bwr_map(hard_pred.astype(np.float))

        image = pv2.images.normalization(image)

        colour_pred = trans * image + (1 - trans) * colour_pred[:, :, :3]
        colour_label = trans * image + (1 - trans) * colour_label[:, :, :3]
        colour_hard = trans * image + (1 - trans) * colour_hard[:, :, :3]

        colour_pred = pv2.images.normalization(colour_pred)
        colour_label = pv2.images.normalization(colour_label)
        colour_hard = pv2.images.normalization(colour_hard)

        rg_map = cm.get_cmap('RdYlGn')
        # rg_map = cm.get_cmap('GnYlRd')
        correct = 1 - np.abs(label.astype(np.float) - prediction[1])
        correct[ignore] = 1
        correct_colour = rg_map(correct)
        # diff_colour = trans * image + (1 - trans) * diff_colour[:, :, :3]

        hard_correct = 1 - np.abs(
            hard_pred.astype(np.float) - label.astype(np.float))
        hard_correct[ignore] = 1
        hdiff_colour = rg_map(hard_correct)

        ax = figure.add_subplot(2, 3, 1)
        ax.set_title('Image')
        ax.axis('off')
        ax.imshow(image)

        ax = figure.add_subplot(2, 3, 4)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(colour_label)

        ax = figure.add_subplot(2, 3, 5)
        ax.set_title('Failure Map')
        ax.axis('off')
        ax.imshow(correct_colour)

        ax = figure.add_subplot(2, 3, 2)
        ax.set_title('Prediction')
        ax.axis('off')
        ax.imshow(colour_pred)

        ax = figure.add_subplot(2, 3, 3)
        ax.set_title('Prediction hard')
        ax.axis('off')
        ax.imshow(colour_hard)

        ax = figure.add_subplot(2, 3, 6)
        ax.set_title('Failure hard')
        ax.axis('off')
        ax.imshow(hdiff_colour)

        return figure


if __name__ == '__main__':
    logging.info("Hello World.")
