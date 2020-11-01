"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import, division, print_function

import copy
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import scipy as scp
from pyvision.evaluation import pretty_printer as pp
from sklearn.metrics import classification_report

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class PVmetric(object):
    """docstring for metric"""
    def __init__(self):
        pass

    def get_pp_names(self, time_unit='s', summary=False):
        raise NotImplementedError

    def get_pp_values(self, ignore_first=True,
                      time_unit='s', summary=False):
        raise NotImplementedError

    def get_pp_dict(self, ignore_first=True, time_unit='s', summary=False):

        names = self.get_pp_names(time_unit=time_unit, summary=summary)
        values = self.get_pp_values(ignore_first=ignore_first,
                                    time_unit=time_unit,
                                    summary=summary)

        return OrderedDict(zip(names, values))


class CombinedMetric(PVmetric):
    """docstring for CombinedMetric"""
    def __init__(self, metriclist):
        super(CombinedMetric, self).__init__()
        self.metriclist = metriclist

    def add(self, idx, *args, **kwargs):
        self.metriclist[idx].add(*args, **kwargs)

    def get_pp_names(self, time_unit='s', summary=False):

        pp_names = []

        for i, metric in enumerate(self.metriclist):
            if metric is None:
                continue
            if i > 0:
                pp_names.append('class_seperator')
            pp_names += metric.get_pp_names(
                time_unit=time_unit, summary=summary)

        return pp_names

    def get_pp_values(self, ignore_first=True,
                      time_unit='s', summary=False):

        pp_values = []

        for i, metric in enumerate(self.metriclist):
            if metric is None:
                continue
            if i > 0:
                pp_values.append(pp.NEW_TABLE_LINE_MARKER)
            pp_values += metric.get_pp_values(
                ignore_first=ignore_first,
                time_unit=time_unit, summary=summary)

        return pp_values

    def get_pp_dict(self, ignore_first=True, time_unit='s', summary=False):

        names = self.get_pp_names(time_unit=time_unit, summary=summary)
        values = self.get_pp_values(ignore_first=ignore_first,
                                    time_unit=time_unit,
                                    summary=summary)

        return OrderedDict(zip(names, values))


class BinarySegMetric(PVmetric):
    """docstring for BinarySegMetric"""
    def __init__(self, thresh=0.5):
        super(BinarySegMetric, self).__init__()
        self.thresh = thresh

        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        self.times = []

    def add(self, prediction, label,
            mask=None, time=None, ignore_idx=None):

        positive = (prediction[0] < self.thresh)

        self.tp += np.sum(positive * label * mask)
        self.fp += np.sum((1 - positive) * label * mask)
        self.fn += np.sum(positive * (1 - label) * mask)
        self.tn += np.sum((1 - positive) * (1 - label) * mask)

        if time is not None:
            self.times.append(time)

    def get_pp_names(self, time_unit='s', summary=False):

        pp_names = []

        pp_names.append("IoU")
        pp_names.append("Precision (PPV)")
        pp_names.append("neg. Prec. (NPV)")
        pp_names.append("Recall (TPR)")
        pp_names.append("Accuracy")
        pp_names.append("Positive")
        if len(self.times) > 0:
            pp_names.append("speed [{}]".format(time_unit))

        return pp_names

    def get_pp_values(self, ignore_first=True,
                      time_unit='s', summary=False):

        pp_values = []

        num_examples = (self.tp + self.fn + self.tn + self.tp)

        iou = self.tp / (self.tp + self.fp + self.fn)

        tp = max(self.tp, 1)
        tn = max(self.tn, 1)

        pp_values.append(iou)
        pp_values.append(tp / (tp + self.fp))
        pp_values.append(tn / (tn + self.fn))
        pp_values.append(tp / (tp + self.fn))
        pp_values.append((tp + tn) / num_examples)
        pp_values.append((tp + self.fp) / num_examples)

        if len(self.times) > 0:
            # pretty printer will multiply all values with 100
            # in order to convert metrics [0, 1] to [0, 100]
            # so times (in seconds) needs to be divided by 100.
            if time_unit == 's':
                pp_values.append(sum(self.times) / len(self.times) / 100)
            elif time_unit == 'ms':
                pp_values.append(10 * sum(self.times) / len(self.times))
            else:
                raise ValueError

        return pp_values

    def get_pp_dict(self, ignore_first=True, time_unit='s', summary=False):

        names = self.get_pp_names(time_unit=time_unit, summary=summary)
        values = self.get_pp_values(ignore_first=ignore_first,
                                    time_unit=time_unit,
                                    summary=summary)

        return OrderedDict(zip(names, values))


class SegmentationMetric(PVmetric):
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

    def add(self, prediction, label,
            mask=None, time=None, ignore_idx=None):
        self.count = self.count + np.sum(mask)
        relevant_classes = set(np.unique(prediction)).union(np.unique(label))

        assert label.shape == prediction.shape

        for cl_id in relevant_classes:

            if cl_id == ignore_idx:
                continue

            pos = label == cl_id
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
            ret_list = copy.copy(self.name_list)
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

    def get_pp_dict(self, ignore_first=True, time_unit='s', summary=False):
        names = self.get_pp_names(time_unit=time_unit, summary=summary)
        values = self.get_pp_values(ignore_first=ignore_first,
                                    time_unit=time_unit,
                                    summary=summary)

        return OrderedDict(zip(names, values))

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


class ClassificationMetric(PVmetric):
    """docstring for ClassificationMetric"""
    def __init__(self, num_classes, name_list=None):
        super().__init__()
        self.num_classes = num_classes
        self.name_list = name_list

        self.predictions = []
        self.labels = []
        self.times = []

    def add(self, prediction, label, mask=True, duration=None):

        if duration is not None:
            self.times.append(duration)

        self.predictions.append(np.argmax(prediction))
        self.labels.append(label)
        self.times.append(duration)

    def get_pp_names(self, time_unit='s', summary=False):
        if not summary:
            ret_list = ["{name:8} [F1]".format(name=name)
                        for name in self.name_list]
            ret_list.append('class_seperator')
        else:
            ret_list = [name for name in self.name_list]

        if len(self.times) > 0:
            ret_list.append("speed [{}]".format(time_unit))

        ret_list.append('accuracy')
        # ret_list.append('precision')
        # ret_list.append('recall')
        # ret_list.append('support')
        ret_list.append('avg f1')

        return ret_list

    def get_pp_values(self, ignore_first=True,
                      time_unit='s', summary=False):

        report = classification_report(
            self.labels, self.predictions, output_dict=True,
            target_names=self.name_list)

        if not summary:
            values = [report[name]['f1-score'] for name in self.name_list]
            values.append(pp.NEW_TABLE_LINE_MARKER)
        else:
            values = [report[name]['f1-score'] for name in self.name_list]

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

        values.append(report['accuracy'])
        # values.append(report['macro avg']['precision'])
        # values.append(report['macro avg']['recall'])
        # values.append(report['macro avg']['support'] / 100)
        values.append(report['macro avg']['f1-score'])

        return values

    def get_pp_dict(self, ignore_first=True, time_unit='s', summary=False):
        names = self.get_pp_names(time_unit=time_unit, summary=summary)
        values = self.get_pp_values(ignore_first=ignore_first,
                                    time_unit=time_unit,
                                    summary=summary)

        return OrderedDict(zip(names, values))


if __name__ == '__main__':
    logging.info("Hello World.")
