"""
The MIT License (MIT)

Copyright (c) 2024 Marvin Teichmann
Email: marvin.teichmann@googlemail.com

The above author notice shall be included in all copies or
substantial portions of the Software.

This file is written in Python 3.8 and tested under Linux.
"""

import gc
import logging
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import scipy as scp

import pyvision
from pyvision.evaluation import pretty_printer as pp

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


class Evaluator:
    def __init__(
        self, conf, model, logger, dataset, SetEvaluator, res_dir=None
    ):
        self.conf = conf
        self.logger = logger
        self.model = model

        self.res_dir = res_dir

        self.create_evaluators(dataset, SetEvaluator)

        self.smoother = OrderedDict(
            {
                key: pyvision.utils.MedianSmoother(
                    self.conf["evaluation"]["num_smoothing_samples"]
                )
                for key in self.evaluators.keys()
            }
        )

        # self.out_path = trainer.out_path

    def create_evaluators(self, dataset, SetEvaluator):

        self.evaluators = OrderedDict()

        self.evaluators["val"] = SetEvaluator(
            self.conf,
            self.model,
            name="val",
            split="val",
            pv_data=dataset,
            subsample=self.conf["evaluation"]["val_subsample"],
            res_dir=self.res_dir,
        )

        self.evaluators["train"] = SetEvaluator(
            self.conf,
            self.model,
            name="train",
            split="train",
            pv_data=dataset,
            subsample=self.conf["evaluation"]["train_subsample"],
            res_dir=self.res_dir,
        )

    def evaluate(self, epoch=None, level="major"):
        self.curr_epoch = epoch
        if self.curr_epoch is None:
            self.curr_epoch = 0
        # self.max_epoch = max_epoch

        # valid_dsc, score_dict, duration = self.validate()
        results = OrderedDict()
        for key, evaluator in self.evaluators.items():
            name = evaluator.name
            logging.info("Evaluating Model on the {} Dataset.".format(name))
            gc.collect()
            start_time = time.time()
            results[key] = evaluator.evaluate(epoch, level=level)
            duration = time.time() - start_time

            logging.info(
                "Finished {} Evaluation of Epoch {:3d} in {:2.2f} minutes.".format(
                    name, self.curr_epoch, duration / 60
                )
            )

        self.log_results(results)

        self._print_summery_string()

    def log_results(self, results):
        first = list(results)[0]
        score_dict = results[first]
        table = pp.TablePrinter(row_names=score_dict.keys())

        for key, result in results.items():
            score_dict = result
            name = self.evaluators[key].name

            values = [i for i in score_dict.values()]
            smoothed = self.smoother[key].update_weights(values)

            table.add_column(smoothed, name=name)
            table.add_column(values, name=name + " (raw)")

        table.print_table()

        if self.curr_epoch:

            for key, result in results.items():
                score_dict = result

                if "class_seperator" in score_dict:
                    del score_dict["class_seperator"]

                self.logger.add_values(
                    value_dict=score_dict,
                    prefix=key,
                    step=self.curr_epoch,
                )

    def _print_summery_string(self):
        epoch = self.curr_epoch
        max_epochs = self.conf["training"]["max_epochs"]

        def median(
            data, weight=self.conf["evaluation"]["num_smoothing_samples"]
        ):
            return np.nanmedian(data[-weight:])

        runname = os.path.basename(os.path.dirname(self.logger.filename))
        if len(runname.split("_")) > 2:
            runname = "{}_{}_{}".format(
                runname.split("_")[0],
                runname.split("_")[1],
                runname.split("_")[2],
            )

        if runname == "":
            runname = "ResNet50"

        out_str = ("Summary:   [{:18}] Epoch: {} / {}").format(
            runname[0:18], epoch, max_epochs
        )

        if "summary" not in self.conf["evaluation"]:
            logging.warning(
                "Summary configuration ('evaluation.summary') not found in 'conf'. Skipping summary."
            )
            logging.info(out_str)
            return

        sconf = self.conf["evaluation"]["summary"]

        if len(self.logger.data) > 0:
            summary_values = []

            # Dynamically calculate metrics based on the configuration
            for prefix, key in zip(sconf["prefixes"], sconf["keys"]):
                metric_key = f"{prefix}\\{key}"
                value = median(self.logger.data[metric_key])
                summary_values.append(value)

            # Build the summary string
            metric_strings = [
                f"{name} {value:.2f}"
                for name, value in zip(sconf["names"], summary_values)
            ]
            out_str += ";   " + "; ".join(metric_strings)

        logging.info(out_str)


if __name__ == "__main__":
    logging.info("Hello World.")
