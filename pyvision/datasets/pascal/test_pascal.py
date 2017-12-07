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


def _assert_data_dir():
    try:
        os.environ['TV_DIR_DATA']
        return True
    except KeyError:
        logging.warning("Data dir not given. Skipping all dataset tests.")
        logging.info("Set $TV_DIR_DATA to perform additional tests.")
        return False
    pass


def test_pascal():
    conf = pinput.default_conf()

    if not _assert_data_dir():
        pass

    if not os.path.exists(os.environ['TV_DIR_DATA'] + "/VOC2012"):
        logging.warning("Dir: {} does not exist."
                        .format(os.environ['TV_DIR_DATA'] + "/VOC2012"))
        logging.info("Skipping pascal voc test.")

    test = pinput.InputProducer(conf)

    next(test)

if __name__ == '__main__':
    logging.info("Hello World.")
