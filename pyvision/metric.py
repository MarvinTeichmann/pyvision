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

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from collections import OrderedDict

from pyvision import pretty_printer as pp


class SegmentationMetric(object):
    """docstring for SegmentationMetric"""
    def __init__(self, num_classes, name_list=None):
        super(SegmentationMetric, self).__init__()
        self.num_classes = num_classes
        self.name_list = name_list

        self.tps = np.zeros([num_classes])
        self.fps = np.zeros([num_classes])
        self.tns = np.zeros([num_classes])
        self.fns = np.zeros([num_classes])

        self.times = []

        self.count = 0

    def add(self, gt, mask, prediction, time=None, ignore_idx=None):
        self.count = self.count + np.sum(mask)
        relevant_classes = set(np.unique(prediction)).union(np.unique(gt))
        for cl_id in relevant_classes:

            if cl_id == ignore_idx:
                continue

            pos = gt == cl_id
            pred = prediction == cl_id

            tp = np.sum(pos * pred * mask)
            fp = np.sum((1 - pos) * pred * mask)
            fn = np.sum(pos * (1 - pred) * mask)
            tn = np.sum((1 - pos) * (1 - pred) * mask)

            assert(tp + fp + fn + tn == np.sum(mask))

            self.tps[cl_id] = self.tps[cl_id] + tp
            self.fps[cl_id] = self.fps[cl_id] + fp
            self.fns[cl_id] = self.fns[cl_id] + fn
            self.tns[cl_id] = self.tns[cl_id] + tn

            if time is not None:
                self.times.append(time)

        return

    def get_iou_dict(self):

        if self.name_list is None:
            name_list = range(self.num_classes)
        else:
            name_list = self.name_list

        ious = self.tps / (self.tps + self.fps + self.fns)
        assert(len(name_list) == len(ious))

        result_dict = OrderedDict(zip(name_list, ious))

        return result_dict

    def compute_miou(self, ignore_first=True):

        ious = self.tps / (self.tps + self.fps + self.fns)

        if ignore_first:
            ious = ious[1:]

        return np.mean(ious)

    def get_accuracy(self, ignore_first=True):

        return np.sum(self.tps) / self.count

    def get_pp_names(self, time_unit='s', summary=False):
        if not summary:
            ret_list = self.name_list
            ret_list.append('class_seperator')
        else:
            ret_list = []

        if len(self.times) > 0:
            ret_list.append("speed [{}]".format(time_unit))

        ret_list.append('accuracy')
        ret_list.append('mIoU')

        return ret_list

    def get_pp_values(self, ignore_first=True,
                      time_unit='s', summary=False):

        ious = self.tps / (self.tps + self.fps + self.fns)

        if not summary:
            values = list(ious)
            values.append(pp.NEW_TABLE_LINE_MARKER)
        else:
            values = []

        if ignore_first:
            ious = ious[1:]

        miou = np.mean(ious)

        if len(self.times) > 0:
            # pretty printer will multiply all values with 100
            # in order to convert metrics [0, 1] to [0, 100]
            # so times (in seconds) needs to be divided by 100.
            if time_unit == 's':
                values.append(sum(self.times) / len(self.times) / 100)
            elif time_unit == 'ms':
                values.append(10 * sum(self.times) / len(self.times))
            else:
                raise ValueError

        values.append(self.get_accuracy(ignore_first=ignore_first))
        values.append(miou)

        return values

    def get_pp_lists(self, ignore_first=True, time_unit='s'):
        crf_dict = self.get_iou_dict()

        crf_dict['class_seperator'] = pp.NEW_TABLE_LINE_MARKER

        if len(self.times) > 0:
            # pretty printer will multiply all values with 100
            # in order to convert metrics [0, 1] to [0, 100]
            # so times (in seconds) needs to be divided by 100.
            if time_unit == 's':
                crf_dict['speed [s]'] = sum(self.times) / len(self.times) / 100
            elif time_unit == 'ms':
                crf_dict['speed [ms]'] = 10 * sum(self.times) / len(self.times)
            else:
                raise ValueError
        crf_dict['accuracy'] = self.get_accuracy(ignore_first=ignore_first)
        crf_dict['mIoU'] = self.compute_miou(ignore_first=ignore_first)

        return crf_dict.keys(), crf_dict.values()

if __name__ == '__main__':
    logging.info("Hello World.")
