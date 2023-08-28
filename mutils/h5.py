"""
The MIT License (MIT)

Copyright (c) 2023 Marvin Teichmann
Email: marvin.teichmann@googlemail.com

The above author notice shall be included in all copies or
substantial portions of the Software.

This file is written in Python 3.8 and tested under Linux.
"""

import os
import sys

import h5py
import numpy as np

import logging


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def save_dict_to_hdf5(filename, data, compression_level=6):
    """
    Save a nested dictionary containing numpy arrays to an HDF5 file.

    Parameters:
    - data: Dictionary to be saved.
    - filename: Name of the HDF5 file.
    - compression_level: Integer from 0 (no compression) to 9 (maximum compression).
    """
    with h5py.File(filename, "w") as f:
        _write_group(f, data, compression_level)


def _write_group(h5file, dictionary, compression_level):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            grp = h5file.create_group(key)
            _write_group(grp, value, compression_level)
        else:
            if (
                isinstance(value, np.ndarray) and value.ndim == 0
            ):  # Check if scalar
                h5file.create_dataset(key, data=value)
            elif isinstance(
                value, (int, float, str)
            ):  # If it's a basic datatype
                h5file.create_dataset(
                    key, data=value
                )  # No compression for these
            else:
                h5file.create_dataset(
                    key,
                    data=value,
                    compression="gzip",
                    compression_opts=compression_level,
                )


def load_dict_from_hdf5(filename):
    """
    Load a nested dictionary containing numpy arrays from an HDF5 file.

    Parameters:
    - filename: Name of the HDF5 file.

    Returns:
    - Loaded dictionary.
    """
    with h5py.File(filename, "r") as f:
        return _read_group(f)


def _read_group(h5file):
    dictionary = {}
    for key in h5file.keys():
        item = h5file[key]
        if isinstance(item, h5py.Dataset):  # check if the item is a dataset
            if item.shape == ():  # check if scalar
                dictionary[key] = item[()]
            else:
                dictionary[key] = item[:]
        elif isinstance(item, h5py.Group):  # check if the item is a group
            dictionary[key] = _read_group(item)
    return dictionary


save = save_dict_to_hdf5
load = load_dict_from_hdf5


if __name__ == "__main__":
    logging.info("Hello World.")
