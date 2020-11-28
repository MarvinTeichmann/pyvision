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


def read_dir(dirname, extension='.txt'):

    files = [file for file in os.listdir(dirname) if file.endswith(extension)]
    files.sort()
    return files
