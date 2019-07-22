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


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('logdirs', nargs='*', type=str,
                        help="directories to plot.")

    parser.add_argument('--default', action='store_true',
                        help="Use default conf of plotter.")

    parser.add_argument('--plotter', type=str,
                        help="Use source found in plotter instead.")

    parser.add_argument('--embed', action='store_true')

    parser.add_argument('--compact', action='store_true')

    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    # args = parser.parse_args()

    return parser


def main(args):

    import matplotlib.pyplot as plt

    if args.plotter is not None:
        plotter = imp.load_source('plotter', args.plotter)
    else:
        if args.logdirs == []:
            logging.info("Usage: pv2 plot logdir1, [logdir2, logdir3, ...]")
            logging.info("Please specify at least one logdir.")
            exit(0)

        main_dir = args.logdirs[0]
        plot_file = os.path.join(main_dir, 'plot.py')
        logging.info("Using plotter defined in: {}".format(plot_file))

        plotter = imp.load_source('plotter', plot_file)

    if args.default:
        config = plotter.default_conf
    else:
        main_dir = args.logdirs[0]
        cfg_file = os.path.join(main_dir, 'config.json')
        logging.info("Using config file: {}".format(cfg_file))
        config = json.load(cfg_file)
        config = config['plotting']

    plotter = plotter.get_pyvision_plotter(config, args.logdirs)

    if args.embed:
        print()
        print()
        plotter.print_keys()
        print()
        if len(plotter.keys()) > 0:
            key_one = [t for t in plotter.keys()][0]
            print("Try 'plotter.plot('%s')' to get started." % key_one)
        print("Tip: Run 'plt.pause(30)' or 'plt.show()' "
              "to unfreeze the Tkinter GUI.")
        print()
        print()
        from IPython import embed
        embed()
        pass
    else:
        plotter.plot_default(compact=args.compact)
        plt.show()

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)

    exit(0)
