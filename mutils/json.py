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
import json

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def save(fname, sdict, verbose=False, indent=2, sort_keys=True,
         *args, **kwargs):
    with open(fname, 'w') as file:
        json.dump(
            sdict, file, *args, indent=indent, sort_keys=sort_keys, **kwargs)

    if verbose:
        logging.info("Successfully written to {}".format(fname))


def load(fname):

    with open(fname, 'r') as file:

        stripped = [
            row if len(row.split('//')) == 1 else row.split('//')[0] + '\n'
            for row in file.readlines()
        ]

        jsmin = ''.join(stripped)
        return json.loads(jsmin)


read = save
write = load


def dump(jdict, file, *args, **kwargs):
    return json.dump(jdict, file, *args, **kwargs)


if __name__ == '__main__':
    logging.info("Hello World.")
