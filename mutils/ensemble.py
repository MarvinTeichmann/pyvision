"""

`ensemble.py`

This module contains utility functions designed to support ensemble learning methodologies in machine learning. 
These utilities focus on feature selection and partitioning, allowing users to effectively create and manipulate 
subsets of features that can be used to train individual models within an ensemble.

The current main function in this module, `generate_feature_partitions`, creates partitions of a given set of features
while ensuring a maximum overlap between any two groups. This allows each model in the ensemble to be trained on 
a distinct set of features, thereby promoting diversity among the base learners. 

As ensemble methods can be highly effective at reducing overfitting and improving model generalization, these utilities 
should prove useful in a variety of machine learning tasks and scenarios. 

Further functions may be added in the future to expand the scope of this module and provide more comprehensive support 
for ensemble learning tasks. 

Functions:
- `sliding_window_partition`
- `generate_feature_partitions`

The MIT License (MIT)

Copyright (c) 2023 Marvin Teichmann
Email: marvin.teichmann@googlemail.com

The above author notice shall be included in all copies or
substantial portions of the Software.

This file is written in Python 3.8 and tested under Linux and Mac OS X.
"""

import os
import sys

import numpy as np
import scipy as scp

import logging
import math

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def sliding_window_partition(all_objects, group_size, overlap_ratio):
    """
    Generates a sequence of overlapping partitions of the input objects
    using a sliding window approach.

    Parameters:
    - all_objects: list of objects to be partitioned.
    - group_size: int, size of each partition.
    - overlap_ratio: int, ratio of overlap between consecutive partitions.

    Returns:
    - A list of lists, each representing a partition of the input objects.
    """
    max_overlap = group_size // overlap_ratio
    step_size = group_size - max_overlap
    num_groups = len(all_objects) // step_size
    groups = []
    objects = all_objects

    for i in range(0, len(objects), step_size):
        next_group = (
            objects[i % len(objects) : i % len(objects) + group_size]
            + objects[: max(0, (i % len(objects) + group_size) - len(objects))]
        )
        if (
            len(groups) > 0
            and len(set(next_group).intersection(set(groups[0]))) > max_overlap
        ):
            break
        groups.append(next_group)
    return groups


def generate_feature_partitions(num_objects, group_size, overlap_ratio):
    """
    Generates a collection of partitions of a set of objects,
    ensuring a maximum overlap between any two partitions.

    Parameters:
    - num_objects: int, number of objects to be partitioned.
    - group_size: int, size of each partition.
    - overlap_ratio: int, ratio of overlap between partitions.

    Returns:
    - A list of lists, each representing a partition of the input objects.
    """
    all_objects = list(range(num_objects))
    groups = []

    # Apply the sliding window to subsets of objects with different divisors
    for i in range(int(math.log(num_objects / group_size, overlap_ratio)) + 1):
        divisor = overlap_ratio**i
        for j in range(int(divisor)):
            subset = [i for i in all_objects if i % divisor == j]
            groups += sliding_window_partition(
                subset, group_size, overlap_ratio
            )

    return groups


def create_partitions_check_overlap(num_objects, group_ratio, overlap_ratio):
    """
    Generates a collection of partitions of a set of objects
    and checks the overlap between partitions.

    Parameters:
    - num_objects: int, number of objects to be partitioned.
    - group_ratio: int, ratio of number of objects to size of each partition.
    - overlap_ratio: int, ratio of overlap between partitions.

    Returns:
    - A list of lists, each representing a partition of the input objects.
    """
    group_size = num_objects // group_ratio
    groups = generate_feature_partitions(
        num_objects, group_size, overlap_ratio
    )

    assert all(len(group) == group_size for group in groups)

    max_overlap = group_size // overlap_ratio
    print(max_overlap)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert len(set(groups[i]).intersection(groups[j])) <= max_overlap

    print(f"Generated {len(groups)} groups")

    return groups


if __name__ == "__main__":
    logging.info("Hello World.")
