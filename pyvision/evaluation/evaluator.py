"""
The MIT License (MIT)

Copyright (c) 2019 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import scipy as scp

import logging
from collections import OrderedDict

import pyvision
from pyvision.evaluation import pretty_printer as pp

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class CombinedEvaluator(object):
    """docstring for MetaEvaluator"""

    def __init__(self, conf, model, imgdir=None):
        self.conf = conf
        self.model = model

        if imgdir is None:
            self.imgdir = os.path.join(model.logdir, "images")
        else:
            self.imgdir = imgdir

        if not os.path.exists(self.imgdir):
            os.mkdir(self.imgdir)

        self.logger = model.logger

        self.evaluators = OrderedDict()

    def evaluate(self, epoch=None, verbose=True, level='minor'):

        metrics = OrderedDict()
        first = list(self.evaluators)[0]

        for name, evaluator in self.evaluators.items():
            logging.info("Evaluating Model on the {} Dataset.".format(name))
            start_time = time.time()
            metrics[name] = evaluator.evaluate(epoch=epoch, level=level)
            dur = time.time() - start_time
            logging.info("Finished {} run in {} minutes.".format(
                name, dur / 60))
            logging.info("")

        if metrics[first] is None:
            logging.info("Metric: {} is None. Stopping evaluation.".format(
                name))
            return

        if verbose:
            # Prepare pretty print

            names = metrics[first].get_pp_names(time_unit="ms", summary=False)
            table = pp.TablePrinter(row_names=names)

            for name, metric in metrics.items():

                values = metric.get_pp_values(
                    time_unit="ms", summary=False, ignore_first=False)
                smoothed = self.evaluators[name].smoother.update_weights(
                    values)

                table.add_column(smoothed, name=name)
                table.add_column(values, name=name + "(raw)")

            table.print_table()

        if epoch is not None:
            for name, metric in metrics.items():
                vdict = metric.get_pp_dict(time_unit="ms", summary=True,
                                           ignore_first=False)
                self.logger.add_values(
                    value_dict=vdict, step=epoch, prefix=name)

            self._print_summery_string(epoch)

        return metrics

    def _print_summery_string(self, epoch):

        max_epochs = self.model.trainer.max_epochs

        def median(data, weight=20):
            return np.median(data[- weight:])

        runname = os.path.basename(self.model.logdir)
        if len(runname.split("_")) > 2:
            runname = "{}_{}_{}".format(runname.split("_")[0],
                                        runname.split("_")[1],
                                        runname.split("_")[2])

        if runname == '':
            runname = "ResNet50"

        out_str = ("Summary:   [{:22}] Epoch: {} / {}").format(
            runname[0:22],
            epoch, max_epochs)

        logging.info(out_str)


if __name__ == '__main__':
    logging.info("Hello World.")
