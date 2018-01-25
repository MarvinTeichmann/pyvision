"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imp
import json
import sys
import argparse
import time

import shutil
from shutil import copyfile

import logging

from datetime import datetime

# import matplotlib.pyplot as plt


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("logdir", type=str,
                        help="configuration file for run.")

    parser.add_argument("--gpus", type=str,
                        help="gpus to use")

    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    # args = parser.parse_args()

    # pvutils.set_gpus_to_use(args)
    return parser


def main(args):

    logdir = args.logdir
    config_file = os.path.join(logdir, 'config.json')
    main_script = os.path.join(logdir, 'model.py')
    source_dir = os.path.join(logdir, 'source')
    add_source = os.path.join(source_dir, 'additional_packages')

    logging.info("Loading Config file: {}".format(config_file))
    config = json.load(open(config_file))
    # TODO Make optional
    sys.path.insert(0, source_dir)
    sys.path.insert(1, add_source)

    m = imp.load_source('model', main_script)

    model = m.create_pyvision_model(conf=config, logdir=logdir)
    model.load_from_logdir()

    start_time = time.time()
    model.fit()
    end_time = (time.time() - start_time) / 3600
    logging.info("Finished training in {} hours".format(end_time))

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)

    exit(0)
