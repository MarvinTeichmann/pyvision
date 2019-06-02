"""
The MIT License (MIT)

Copyright (c) 2018 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

import matplotlib.pyplot as plt

from pyvision.logger import Logger
from pprint import pprint

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from threading import Thread

default_conf = {
    "defaults": {
        "plot": ["mIoU", "accuracy", "mIoU"],
        "prefixes": ["val", "val", "train"],
        "titles": ["Validation mIoU", "Validation Accuracy", "Train mIoU"],
    }
}


def get_pyvision_plotter(conf, logdirs, names=None):

    summary_name = 'summary.log.hdf5'

    filenames = [os.path.join(name, summary_name) for name in logdirs]
    loggers = [Logger().load(file) for file in filenames]
    if names is None:
        names = [os.path.basename(name) for name in logdirs]

    return Plotter(loggers, names, conf)


class Plotter(object):
    """docstring for Plotter"""

    def __init__(self, loggers, names, config):
        super(Plotter, self).__init__()
        self.loggers = loggers
        self.names = names
        self.steps = [logger.steps for logger in loggers]

        self.config = config

    def keys(self):
        return self.loggers[0].data.keys()

    def print_keys(self):
        logging.info("Available keys are:")
        pprint([t for t in self.keys()])
        return
        string = None
        for i, key in enumerate(self.keys()):
            if i % 5 == 0:
                if string is not None:
                    logging.info(string)
                string = "    "
            string = string + key + 4 * " "
        logging.info(string)

    def plot_default(self, compact=False):

        default = self.config['defaults']

        iterator = zip(default['plot'],
                       default['prefixes'],
                       default['titles'])

        for key, prefix, title in iterator:
            self.plot(key, prefix=prefix, title=title)
            if compact:
                break

    def medianize(self, data, weight=20, unbiased=True):
        if unbiased:
            medianized = [np.median(data[max(i - weight, 0): i + min(i, weight) + 1]) # NOQA
                          for i in range(len(data))]
        else:
            medianized = [np.median(data[max(i - weight, 0): i + 1])
                          for i in range(len(data))]
        return medianized

    def plot(self, key, prefix=None, title=None, smoothed=True, percent=True):

        if prefix is not None:
            key = prefix + "\\" + key

        if title is None:
            title = key

        data = [logger.data[key] for logger in self.loggers]

        if smoothed:
            self.plot_data(self.steps, data, names=self.names,
                           title=title, percent=percent)
        else:
            self.plot_data(self.steps, data, names=self.names,
                           title=title, marker=" ", linestyle='-',
                           plot_smoothed=False, percent=percent)

    def plot_miuo(self):
        self.plot(key="mIoU", prefix="val", title="Validation mIoU")

    def plot_accuracy(self):
        self.plot(key="accuracy", prefix="val",
                      title="Validation Accuracy")

    def plot_train_miuo(self):
        self.plot(key="mIoU", prefix="train", title="Train mIoU")

    def reduce_data(self, epoch):
        for logger in self.loggers:
            for key in logger.data.keys():
                logger.data[key] = logger.data[key][:epoch]
            logger.steps = logger.steps[:epoch]
        self.steps = [logger.steps for logger in self.loggers]

    def get_logger(self):
        return self.loggers[0]

    def plot_data(self, steps, plot_data, names,
                  plot_smoothed=True, title=None,
                  marker=".", linestyle=' ', annotate=True, percent=True,
                  ax=None, show=True, show_legend=True):

        if len(steps) != len(names):
            steps = [steps for i in range(len(names))]

        assert(len(names) == len(plot_data))

        # plt.rcParams.update({'font.size': 14})
        # fig, ax = plt.subplots(figsize=(6.4, 4.8))
        if ax is None:
            fig, ax = plt.subplots()

        iterator = zip(plot_data, names, steps)

        for i, (data, name, mysteps) in enumerate(iterator):

            if percent:
                p_data = [100 * d for d in data]
            else:
                p_data = data

            smoothed = self.medianize(p_data, unbiased=True)

            color = 'C{}'.format(i)

            # Do plotting
            ax.plot(mysteps, p_data, marker=marker, linestyle=linestyle,
                    label=name + " (raw)", color=color)

            if plot_smoothed:
                ax.plot(mysteps, smoothed,
                        label=name + " (smooth)", color=color)

                if annotate:
                    self._do_annotation(ax, color, smoothed, mysteps, i)

            if title is not None:
                ax.set_title(title)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score [%]')
            # ax.legend()
            if show_legend:
                ax.legend(loc=0)

        if show is True:
            plt.pause(0.01)

    def _do_annotation(self, ax, color, data, steps, iter):
        iter2 = 0
        for x_ann, y_ann, name in self.get_annotation_points(data, steps):

            ax.plot([x_ann], [y_ann], 'o', color='red')

            ax.annotate('{}: {:.2f}'.format(name, y_ann),
                        xy=(x_ann, y_ann),
                        xytext=(-20 + 20 * iter2, 20 + 10 * iter2 + 20 * iter),
                        textcoords='offset points',
                        color=color, size=10,
                        horizontalalignment='center',
                        arrowprops=dict(arrowstyle="->",
                                        alpha=0.75, color=color))

            iter2 = iter2 + 1

    def get_annotation_points(self, smoothed, steps):
        ymax = np.max(smoothed)
        idmax = np.argmax(smoothed)
        xmax = steps[idmax]

        p1 = (xmax, ymax, 'max')

        if ymax == smoothed[-1]:
            return [p1]

        ymin = np.min(smoothed[idmax:])
        xmin = np.argmin(smoothed[idmax:])

        p2 = (xmax + steps[xmin], ymin, 'min')

        if ymin == smoothed[-1]:
            return [p1, p2]

        ylast = smoothed[-1]
        xlast = len(smoothed) - 1
        p3 = (steps[xlast], ylast, 'last')

        return [p1, p2, p3]

if __name__ == '__main__':
    logging.info("Hello World.")
