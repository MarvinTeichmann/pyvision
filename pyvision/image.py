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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def normalize(img, whitening=False):

    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    if whitening:

        std_cap = 1.0 / np.sqrt((len(img)))

        adjusted_stddev = max(np.std(img), std_cap)
        img = (img - np.mean(img)) / adjusted_stddev

    return img


if __name__ == '__main__':
    logging.info("Hello World.")
