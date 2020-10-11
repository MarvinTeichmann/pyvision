"""
The MIT License (MIT)
Copyright (c) 2020 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import pandas as pd

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def describe(array):
    df = pd.DataFrame(data=array.flatten())
    return df.describe()


if __name__ == '__main__':
    logging.info("Hello World.")
