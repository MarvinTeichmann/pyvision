"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imp
import sys
import argparse
import time

import shutil
from shutil import copyfile

import logging

from datetime import datetime

from .. import utils as pvutils

from mutils import json

# import matplotlib.pyplot as plt


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('logdir', nargs='*', type=str,
                        help="directories to plot.")

    parser.add_argument('--default', action='store_true',
                        help="Use default conf of plotter.")

    parser.add_argument('--eval', type=str,
                        help="Use source found in plotter instead.")

    parser.add_argument('--embed', action='store_true')

    parser.add_argument('--compact', action='store_true')

    parser.add_argument("--gpus", type=str,
                        help="gpus to use")

    parser.add_argument("--eval_file", type=str,
                        default="eval_out.log")

    parser.add_argument("--name", type=str,
                        default='eval_out')

    parser.add_argument("--checkpoint", type=str,
                        default=None)

    parser.add_argument("--data", type=str,
                        default=None)

    parser.add_argument("--level", type=str,
                        default="mayor")

    parser.add_argument('--sys_packages', action='store_true',
                        help='Use system source for all packages.')
    parser.add_argument('--add_packages', action='store_true',
                        help='Use local source for additional_packages.')

    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    # args = parser.parse_args()

    return parser


def main(args):

    pvutils.set_gpus_to_use(args)

    logdir = args.logdir[0]
    logdir = os.path.realpath(logdir)
    config_file = os.path.join(logdir, 'config.json')
    main_script = os.path.join(logdir, 'model.py')

    source_dir = os.path.join(logdir, 'source')
    add_source = os.path.join(source_dir, 'additional_packages')

    logging.info("Loading Config file: {}".format(config_file))
    config = json.load(config_file)

    if args.add_packages:
        sys.path.insert(0, add_source)
    if not args.sys_packages:
        sys.path.insert(0, source_dir)

    imgdir = os.path.join(logdir, args.name)
    # Create an output log file
    logfile = os.path.join(imgdir, args.eval_file)
    logging.info("All output will be written to: {}".format(logfile))
    pvutils.create_filewrite_handler(logfile, mode='a')

    m = imp.load_source('model', main_script)

    model = m.create_pyvision_model(conf=config, logdir=logdir)

    if args.checkpoint is not None:
        model.load_from_logdir(ckp_name=args.checkpoint)
    else:
        model.load_from_logdir()

    logging.info("Model loaded. Starting evaluation.")

    if args.eval is None:
        pveval = model.pv_evaluator.get_pyvision_evaluator(
            config, model, imgdir=imgdir, dataset=args.data)
    else:
        evaluator = imp.load_source('evaluator', args.eval)
        pveval = evaluator.get_pyvision_evaluator(
            config, model, imgdir=imgdir, dataset=args.data)

    start_time = time.time()
    pveval.evaluate(level=args.level)
    end_time = (time.time() - start_time) / 60
    logging.info("Finished training in {} minutes".format(end_time))


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)

    exit(0)
