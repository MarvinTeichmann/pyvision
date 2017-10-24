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

from pyvision.datasets import pascal
from pyvision.datasets.pascal import input_producer as pinput


def test_pascal():
    conf = pinput.default_conf()

    data_dir = '/home/mifs/mttt2/cvfs/DATA'
    test = pinput.InputProducer(conf, data_dir)

    next(test)

if __name__ == '__main__':
    logging.info("Hello World.")
