"""
The MIT License (MIT)

Copyright (c) 2021 Marvin Teichmann
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

import random

import shutil

from pathlib import Path


from sklearn.model_selection import KFold

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def do_kfold(index, fold=0, folds=5, seed=42, **kwargs):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train_split, val_split = [i for i in kf.split(index)][fold]

    train_index = [item for i, item in enumerate(index) if i in train_split]

    val_index = [item for i, item in enumerate(index) if i in val_split]

    return train_index, val_index


class RamDiskManager:
    def __init__(
        self, slow_root, fast_root="/dev/shm", copy=1.0, recopy=0.02, buffer=1
    ):
        self.slow_root = slow_root
        self.fast_root = fast_root
        self.copy = copy
        self.recopy = recopy
        self.buffer = buffer

    def get_path(self, path, copy=None):
        target = os.path.join(self.fast_root, path)
        source = os.path.join(self.slow_root, path)
        lock_file = os.path.join(self.fast_root, path + ".lock")

        assert target != path, (
            f"Absolute path given. Path should be a relative path"
            f"{self.slow_root}"
        )

        if os.path.exists(target):
            if not os.path.exists(lock_file):
                # TODO: Consider checking age of lock
                if random.random() < self.recopy:
                    return self.copy_to_ram(path)
                return target
            else:
                return source

        if copy is None:
            copy = self.copy

        if random.random() < copy:
            free_space = shutil.disk_usage(self.fast_root)[2] / 1024**3
            if free_space < self.buffer:
                return source

            return self.copy_to_ram(path)
        else:
            return source

    def copy_to_ram(self, path):
        target = os.path.join(self.fast_root, path)
        source = os.path.join(self.slow_root, path)
        lock_file = os.path.join(self.fast_root, path + ".lock")

        if not os.path.exists(os.path.dirname(lock_file)):
            os.makedirs(os.path.dirname(lock_file))

        if not os.path.exists(lock_file):
            try:
                Path(lock_file).touch()
                shutil.copy2(source, target)
                os.remove(lock_file)
                return target
            except:  # NOQA
                return source
        else:
            return source

    def delete_from_ram(self, path):
        target = os.path.join(self.fast_root, path)
        lock_file = os.path.join(self.fast_root, path + ".lock")

        Path(lock_file).touch()
        if os.path.exists(target):
            os.remove(target)

        os.remove(lock_file)


if __name__ == "__main__":
    rmd = RamDiskManager(slow_root="/")
    this_file = os.path.realpath(__file__)[1:]
    rmd.get_path(this_file, copy=True)
