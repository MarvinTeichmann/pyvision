"""
The MIT License (MIT)

Copyright (c) 2022 Marvin Teichmann
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

from time import sleep

import pyvision
import pyvision.logger
import pyvision.evaluation.distributions as dist

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def create_pyvision_model(conf, logdir, init_training=True, debug=False):
    model = PVControll(
        conf=conf, logdir=logdir, debug=debug, init_training=init_training
    )
    return model


class PVControll:
    def __init__(self, conf, logdir="tmp", debug=False, init_training=False):

        self.conf = conf
        self.logdir = logdir

        # flag: --debug
        self.debug = debug
        self.init_training = init_training

        summary_file = os.path.join(logdir, "summary.log.hdf5")
        self.logger = pyvision.logger.Logger(filename=summary_file)

        if debug:
            self.set_conf_debug()

    def set_conf_debug(self):
        """
        Set the model into debug mode. Reduce epochs and complexity of model.
        """

        self.conf["model"]["keyword"] = "Debug?"
        self.conf["training"]["num_epochs"] = 4
        self.conf["training"]["steps_per_epoch"] = 3
        self.conf["training"]["default_device"] = "cpu"

    def load_from_logdir(self, logdir=None, ckp_name=None):

        if logdir is None:
            logdir = self.logdir

        if ckp_name is None:
            checkpoint_name = os.path.join(logdir, "checkpoint.pth.tar")
        else:
            checkpoint_name = os.path.join(logdir, ckp_name)

        if not os.path.exists(checkpoint_name):
            logging.info("No checkpoint file found. Train from scratch.")
            return

    def evaluate(self, epoch=None, verbose=True, level="minor", dataset=None):
        keyword = self.conf["model"]["keyword"]
        logging.info("Evaluating Epoch {}...".format(epoch))
        sleep(1)

        rng_error = 1 / (epoch + 1) * dist.skewed_normal(0.2)
        performance = 1 - rng_error

        logging.info(
            "    Model says: {} With performance {:0.4f}".format(
                keyword, performance
            )
        )

        return performance

    def _finalize_epoch(self, epoch):
        """Run end of epoch stuff."""

        result = self.evaluate(epoch)

        # Log results
        self.logger.init_step(epoch)
        self.logger.add_value(epoch, "Epoch", epoch)
        self.logger.add_value(result, "Performance", epoch, prefix="Val")
        self.logger.add_value(
            result ** 2, "PerformanceSquare", epoch, prefix="Val"
        )
        self.logger.save()

    def fit(self, max_epochs=None):

        num_epochs = self.conf["training"]["num_epochs"]

        logging.info("Training for {} epochs".format(num_epochs))

        for epoch in range(num_epochs):
            logging.info(
                "Training epoch {:2d} / {:2d} ... ".format(epoch, num_epochs)
            )
            sleep(2)

            self._finalize_epoch(epoch)


if __name__ == "__main__":
    logging.info("Hello World.")
