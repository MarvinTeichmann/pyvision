"""
The MIT License (MIT)

Copyright (c) 2021 Marvin Teichmann
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

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


if __name__ == "__main__":
    logging.info("Hello World.")
