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

import logging
import json
import shutil

import time
import traceback

from multiprocessing import Process

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


if __name__ == '__main__':
    logging.info("Hello World.")


def robust_training(model, restarts=5, subprocess=False):

    if not subprocess:
        robust_training_exceptions(model, restarts=restarts)
    else:
        robust_training_subprocess(model, restarts=restarts)


def robust_training_exceptions(model, restarts=5):
    start_time = time.time()

    crash_count = 0
    crashed_epoch = 0
    crash_epoch_count = 0

    model.epoch = 0

    while True:

        try:
            model.load_from_logdir()
            logging.info("Starting training at epoch: {}".format(model.epoch))

            model.fit()

            # p = Process(target=model.fit)
            # p.start()
            # p.join()
            break
        except KeyboardInterrupt:
            break
        except:
            # logging.info("Error: {}".format(sys.exc_info()[0]))
            traceback.print_exc()
            print()

            crash_count += 1
            logging.warning("Training was KILLED, count: {}".format(
                crash_count))

            if crashed_epoch >= model.epoch:
                crash_epoch_count += 1
                if crash_epoch_count >= restarts:
                    logging.info(
                        "Model crashed {} times at epoch {}. "
                        "Stopping training.".format(
                            restarts + 1, crashed_epoch))
                    break
            else:
                crashed_epoch = model.epoch
                crash_epoch_count = 0

    end_time = (time.time() - start_time) / 3600
    logging.info("Finished training in {} hours".format(end_time))


def robust_training_subprocess(model, restarts=5):
    start_time = time.time()

    crash_count = 0
    crashed_epoch = 0
    crash_epoch_count = 0

    model.epoch = 0

    logging.warning("Run training in a seperate process."
                    " Make sure that your model supports multiprocessing or"
                    " deactivate robust training.")

    while True:

        model.load_from_logdir()
        logging.info("Starting training at epoch: {}".format(model.epoch))

        p = Process(target=model.fit)
        p.start()
        p.join()

        if p.exitcode == 0:
            break
        else:
            # logging.info("Error: {}".format(sys.exc_info()[0]))
            # traceback.print_exc()

            crash_count += 1
            logging.warning("Training was KILLED, count: {}".format(
                crash_count))

            if crashed_epoch >= model.epoch:
                crash_epoch_count += 1
                if crash_epoch_count >= restarts:
                    logging.info(
                        "Model crashed {} times at epoch {}. "
                        "Stopping training.".format(
                            restarts + 1, crashed_epoch))
                    break
            else:
                crashed_epoch = model.epoch
                crash_epoch_count = 0

    end_time = (time.time() - start_time) / 3600
    logging.info("Finished training in {} hours".format(end_time))


def set_gpus_to_use(args, gpus=None):
    """Set the gpus to use."""
    if gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    if args.gpus is None:
        if 'TV_USE_GPUS' in os.environ:
            if os.environ['TV_USE_GPUS'] == 'force':
                logging.error('Please specify a GPU.')
                logging.error('Usage {} --gpus <ids>'.format(sys.argv[0]))
                exit(1)
    else:
        logging.info("GPUs are set to: %s", args.gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


def create_filewrite_handler(logging_file, mode='w'):
    """
    Create a filewriter handler.

    A copy of the output will be written to logging_file.

    Parameters
    ----------
    logging_file : string
        File to log output

    Returns
    -------
    The filewriter handler
    """
    target_dir = os.path.dirname(logging_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filewriter = logging.FileHandler(logging_file, mode=mode)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-3s %(levelname)-3s %(message)s')
    filewriter.setLevel(logging.INFO)
    filewriter.setFormatter(formatter)
    logging.getLogger('').addHandler(filewriter)
    return filewriter


def initialize_output_dir(cfg, cfg_file, output_dir, do_logging=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        logging.warning("Path exists: {}".format(output_dir))
        logging.warning("Potentially overwriting existing model.")

    target_file = os.path.join(output_dir, 'conf.json')
    with open(target_file, 'w') as outfile:
        json.dump(cfg, outfile, indent=2, sort_keys=True)

    base_path = os.path.dirname(os.path.realpath(cfg_file))

    for dir_name in cfg['copy_dirs']:
        src = os.path.join(base_path, dir_name)
        src = os.path.realpath(src)

        name = os.path.basename(dir_name)
        dst = os.path.join(output_dir, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Creating an additional logging saving the console outputs
    # into the training folder
    # if do_logging:
    #    logging_file = os.path.join(output_dir, "output.log")
    #    create_filewrite_handler(logging_file)

    return


class ExpoSmoother():
    """docstring for expo_smoother"""
    def __init__(self, decay=0.9):
        self.weights = None
        self.decay = decay

    def update_weights(self, l):
        if self.weights is None:
            self.weights = np.array(l)
            return self.weights
        else:
            dec = self.decay * self.weights
            self.weights = dec + (1 - self.decay) * np.array(l)
            return self.weights

    def get_weights(self):
        return self.weights.tolist()


class MedianSmoother():
    """docstring for expo_smoother"""
    def __init__(self, num_entries=50):
        self.weights = None
        self.num = 50

    def update_weights(self, l):
        l = np.array(l).tolist()
        if self.weights is None:
            self.weights = [[i] for i in l]
            return [np.median(w[-self.num:]) for w in self.weights]
        else:
            for i, w in enumerate(self.weights):
                w.append(l[i])
            if len(self.weights) > 20 * self.num:
                self.weights = [w[-self.num:] for w in self.weights]
            return [np.median(w[-self.num:]) for w in self.weights]

    def get_weights(self):
        return [np.median(w[-self.num:]) for w in self.weights]
