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

import warnings
import deepdish as dd

import logging
from tables.exceptions import NaturalNameWarning

import pickle

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class Logger():
    def __init__(self, filename="summary.log.hdf5"):
        self.data = {}
        self.steps = []
        self.filename = filename

    def init_step(self, step):
        self.steps.append(step)
        if len(self.steps) > 1:
            # Check that step size is constant.
            assert(self.steps[-1] - self.steps[-2] ==
                   self.steps[1] - self.steps[0])

    def add_value(self, value, name, step):
        # TODO: make more pythonic / permissive
        assert(self.steps[-1] == step)
        if len(self.steps) == 1:
            self.data[name] = [value]
        else:
            self.data[name].append(value)
            assert(len(self.data[name]) == len(self.steps))

    def add_values(self, value_dict, step, prefix=None):
        for name, value in value_dict.items():
            if prefix is not None:
                name = prefix + "\\" + name
            self.add_value(value, name, step)

    def save(self, filename, hdf=False):
        if filename is None:
            assert(self.filename is not None)
            filename = self.filename

        save_dict = {'data': self.data,
                     'steps': self.steps}

        if hdf:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NaturalNameWarning)
                dd.io.save(filename, save_dict)
        else:
            pickle.dump(save_dict, open(filename, 'wb'))

    def load(self, filename=None):

        if filename is None:
            filename = self.filename

        try:
            load_dict = pickle.load(open(filename, 'rb'))
        except pickle.UnpicklingError:
            load_dict = dd.io.load(filename)
        self.data = load_dict['data']
        self.steps = load_dict['steps']
        return self

    def reduce_step(self, step):
        reduced_data = {}
        assert(step >= 0)
        assert(step <= len(self.steps))
        for key, value in self.data.items():
            reduced_data[key] = value[step]
        return reduced_data

    def discard_data(self, step):
        reduced_data = {}
        assert(step >= 0)
        assert(step <= len(self.steps))
        for key, value in self.data.items():
            reduced_data[key] = value[0:step]
        self.data = reduced_data
        self.steps = self.steps[0:step]
        return


if __name__ == '__main__':
    logging.info("Hello World.")
