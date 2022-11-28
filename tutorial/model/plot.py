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

from pyvision.logger import Logger
from pyvision.plotter import Plotter as PVPlotter


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def get_pyvision_plotter(conf, logdirs, names=None):

    summary_name = "summary.log.hdf5"

    filenames = [os.path.join(name, summary_name) for name in logdirs]
    loggers = [Logger().load(file) for file in filenames]
    if names is None:
        names = [os.path.basename(name) for name in logdirs]

    return Plotter(loggers, names, conf)


class Plotter(PVPlotter):
    def __init__(self, loggers, names, conf):
        super().__init__(loggers, names, conf)


if __name__ == "__main__":
    logging.info("Hello World.")
