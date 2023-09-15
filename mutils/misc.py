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


def list_files_by_extension(dirname, extension=".npz"):
    """
    List all files in a directory with a given file extension and sort them
    alphabetically.

    Args:
    - dirname (str): The path to the directory to scan.
    - extension (str): The file extension to filter by. Default is '.npz'.

    Returns:
    - list: A sorted list of files with the specified file extension.
    """

    if not os.path.exists(dirname):
        raise ValueError(f"The provided directory '{dirname}' does not exist.")

    files = [file for file in os.listdir(dirname) if file.endswith(extension)]
    files.sort()

    return files


read_dir = list_files_by_extension
