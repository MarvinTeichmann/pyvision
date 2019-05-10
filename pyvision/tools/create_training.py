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

from pyvision import utils as pvutils
from pyvision import organizer as pvorg

from time import sleep

# import matplotlib.pyplot as plt


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("config", type=str,
                        help="configuration file for run.")

    parser.add_argument("--bench", type=str, default='debug',
                        help="Subfolder to store all runs.")

    parser.add_argument("--name", type=str,
                        help="Name for the run.")

    parser.add_argument("--gpus", type=str,
                        help="gpus to use")

    parser.add_argument('--notimestamp', action='store_false',
                        dest='timestamp', help="Run in Debug mode.",
                        default=True)

    parser.add_argument('--wait', type=int,
                        help="Wait till gpus are available.")

    parser.add_argument('--debug', action='store_true',
                        help="Run in Debug mode.")

    parser.add_argument('--train', action='store_true',
                        help="Do training. \n"
                        " Default: False; Only Initialize dir.")

    parser.add_argument('--restarts', type=int,
                        default=0,
                        help="Restart training [num] times when crashed.")

    parser.add_argument('--subprocess', action='store_true',
                        help="Run training as subprocess, allowing to restart"
                             " after segmentation faults.")

    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    # args = parser.parse_args()

    return parser


def main(args):

    pvutils.set_gpus_to_use(args)

    logging.info("Loading Config file: {}".format(args.config))
    config = json.load(open(args.config))

    logdir = pvorg.get_logdir_name(
        project=config['pyvision']['project_name'],
        bench=args.bench, cfg_file=args.config, prefix=args.name,
        timestamp=args.timestamp)

    pvorg.init_logdir(config, args.config, logdir)

    logging.info("Model initialized in: ")
    logging.info(logdir)

    if args.wait:
        import GPUtil
        while GPUtil.getGPUs()[0].memoryUtil > 0.1:
            logging.info("GPU 0 is beeing used.")
            GPUtil.showUtilization()
            sleep(60)

    if args.debug or args.train:

        sfile = config['pyvision']['entry_point']

        model_file = os.path.realpath(os.path.join(
            os.path.dirname(args.config), sfile))

        assert(os.path.exists(model_file))

        m = imp.load_source('model', model_file)

        mymodel = m.create_pyvision_model(config, logdir=logdir)

        if args.debug:
            restarts = 0
        else:
            restarts = args.restarts

        pvutils.robust_training(mymodel, restarts=restarts,
                                subprocess=False)

        # Do forward pass
        # img_var = Variable(sample['image']).cuda() # NOQA
        # prediction = mymodel(img_var)
    else:
        logging.info("Initializing only mode. [Try train.py --train ]")
        logging.info("To start training run:")
        logging.info("    pv2 train {} --gpus".format(logdir))

    exit(0)

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)

    exit(0)
