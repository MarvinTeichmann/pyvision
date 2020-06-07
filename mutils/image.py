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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def normalize(img, whitening=False, verbose=True):

    img = img.astype(np.float32)

    if np.abs(np.max(img) - np.min(img)) < 1e-10:
        img[:] = 0
        if verbose:
            logging.warning("Image is Constant.")
        return img

    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    if whitening:

        std_cap = 1.0 / np.sqrt((len(img)))

        adjusted_stddev = max(np.std(img), std_cap)
        img = (img - np.mean(img)) / adjusted_stddev

    return img


def plot(*args, backend='Qt5agg'): # NOQA

    matplotlib.use(backend)
    num_images = len(args)

    fig, axes = plt.subplots(1, num_images)

    for ax, img in zip(axes, args):
        ax.imshow(img)

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    logging.info("Hello World.")
