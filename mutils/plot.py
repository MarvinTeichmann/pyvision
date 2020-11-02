"""
The MIT License (MIT)
Copyright (c) 2020 Marvin Teichmann
"""

from __future__ import absolute_import, division, print_function

import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as scp

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def plot_corrolation(x, y, x_name=None, y_name=None, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))

    ax.scatter(x, y, **kwargs)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    slope, intercept, corrcoef, p, stderr = scp.stats.linregress(x, y)
    ax.set_title("Corrolation: {:2.2f}".format(100 * corrcoef))
    ax.plot(x, x * slope + intercept, color='orange')


if __name__ == '__main__':
    logging.info("Hello World.")
