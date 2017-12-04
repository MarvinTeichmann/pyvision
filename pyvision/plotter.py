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

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def smooth_median(data, weight=20):
    medianized = [np.median(data[max(i - weight, 0): i + 1])
                  for i in range(len(data))]

    return medianized


def get_annotation_points(smoothed):
    ymax = np.max(smoothed)
    xmax = np.argmax(smoothed)

    p1 = (xmax, ymax, 'max')

    if ymax == smoothed[-1]:
        return [p1]

    ymin = np.min(smoothed[xmax:])
    xmin = np.argmin(smoothed[xmax:])

    p2 = (xmax + xmin, ymin, 'min')

    if ymin == smoothed[-1]:
        return [p1, p2]

    ylast = smoothed[-1]
    xlast = len(smoothed) - 1
    p3 = (xlast, ylast, 'last')

    return [p1, p2, p3]


def plot_dot_smoothed(steps, plot_data, names, plot_smoothed=True):

    if len(steps) != len(names):
        steps = [steps for i in range(len(names))]

    assert(len(names) == len(plot_data))

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(16, 10))

    for i, (data, name, mysteps) in enumerate(zip(plot_data, names, steps)):

        p_data = [100 * d for d in data]

        smoothed = smooth_median(p_data)

        color = 'C{}'.format(i)

        # Do plotting
        ax.plot(mysteps, p_data, marker=".", linestyle=' ',
                label=name + " (raw)", color=color)

        if plot_smoothed:
            ax.plot(mysteps, smoothed,
                    label=name + " (smooth)", color=color)

        off_1 = 0.5
        # off_2 = 2 + 0.5 * i
        off_2 = 2
        for x_ann, y_ann, name in get_annotation_points(smoothed):
            off_1 = - off_1
            off_2 = - off_2

            ax.plot([x_ann], [y_ann], 'o', color='red')

            ax.annotate('{}: {:.2f}'.format(name, y_ann),
                        xy=(x_ann, y_ann - off_1),
                        xytext=(x_ann, y_ann - off_2), color=color,
                        size=10,
                        horizontalalignment='center',
                        arrowprops=dict(arrowstyle="->"))

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score [%]')
        ax.legend(loc=0)

if __name__ == '__main__':
    logging.info("Hello World.")
