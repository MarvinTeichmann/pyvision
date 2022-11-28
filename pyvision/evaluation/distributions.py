"""
The MIT License (MIT)

Copyright (c) 2022 Marvin Teichmann
Email: marvin.teichmann@googlemail.com

The above author notice shall be included in all copies or
substantial portions of the Software.

This file is written in Python 3.8 and tested under Linux.
"""

import os
import sys

import numpy as np
import scipy as scp

import logging

import random

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def skewed_normal(std=0, mean=1):

    diff = random.normalvariate(0, std)

    if diff < 0:
        factor = 1 / (1 - diff)
    else:
        factor = mean + diff

    factor *= mean

    return factor


def truncated_normal(mean=0, std=0, lower=-0.5, upper=0.5):

    while True:

        factor = random.normalvariate(mean, std)

        if factor > lower and factor < upper:
            break

    return factor


if __name__ == "__main__":
    logging.info("Hello World.")
