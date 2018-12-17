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

from .. import utils as pvutils

from time import sleep

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

    parser.add_argument('--wait', type=int,
                        help="Wait till gpus are available.")

    parser.add_argument('--restarts', type=int,
                        default=5,
                        help="Restart training [num] times when crashed.")

    parser.add_argument('--subprocess', action='store_true',
                        help="Run training as subprocess, allowing to recover"
                             "from segmentation faults.")

    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    # args = parser.parse_args()

    return parser


def main(args):

    pvutils.set_gpus_to_use(args)

    logdir = args.logdir
    logdir = os.path.realpath(logdir)
    config_file = os.path.join(logdir, 'config.json')
    main_script = os.path.join(logdir, 'model.py')
    source_dir = os.path.join(logdir, 'source')
    add_source = os.path.join(source_dir, 'additional_packages')

    logging.info("Loading Config file: {}".format(config_file))
    config = json.load(open(config_file))
    # TODO Make optional
    sys.path.insert(0, source_dir)
    sys.path.insert(1, add_source)

    # Create an output log file
    logfile = os.path.join(logdir, 'output.log')
    logging.info("All output will be written to: {}".format(logfile))
    pvutils.create_filewrite_handler(logfile, mode='a')

    if args.wait is not None:
        import GPUtil
        gpu_id = args.wait
        while GPUtil.getGPUs()[gpu_id].memoryUtil > 0.2:
            logging.info("GPU {} is beeing used.".format(gpu_id))
            GPUtil.showUtilization()
            sleep(60)

    m = imp.load_source('model', main_script)

    model = m.create_pyvision_model(conf=config, logdir=logdir)

    pvutils.robust_training(model, subprocess=args.subprocess,
                            restarts=args.restarts)

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)

    exit(0)
