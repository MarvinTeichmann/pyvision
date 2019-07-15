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

import json
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def save(fname, sdict, indent=2, sort_keys=True, verbose=False,
         *args, **kwargs):
    with open(fname, 'w') as file:
        json.dump(
            sdict, file, *args, indent=indent, sort_keys=sort_keys, **kwargs)

    if verbose:
        logging.info("Successfully written to {}".format(fname))


def load(fname):
    with open(fname, 'r') as file:
        return json.load(open(file, 'r'))


if __name__ == '__main__':
    logging.info("Hello World.")
