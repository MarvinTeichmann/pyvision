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

from typing import List

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
    step_size = int(group_size - max_overlap)
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
            if len(set(groups[i]).intersection(groups[j])) > max_overlap:
                logging.error(
                    "Overlap: {} (Max: {})".format(
                        len(set(groups[i]).intersection(groups[j])),
                        max_overlap,
                    )
                )
            assert len(set(groups[i]).intersection(groups[j])) <= max_overlap

    print(f"Generated {len(groups)} groups")

    return groups


def dyadic_partitions(
    num_objects: int, num_partitions: [int] = None
) -> List[List[int]]:
    """
    Implements the dyadic "take/skip blocks" strategy:

      Level 0: A0 = first half,             A0c = complement
      Level 1: A1 = 1st quarter + 3rd,      A1c = complement
      Level 2: A2 = 1st,3rd,5th,7th eighths A2c = complement
      ...

    Args:
      num_objects (N): number of objects labeled 0..N-1
      num_partitions: optional total number of partitions (lists) to return.
                      If provided, generation stops once that many lists
                      have been produced (or earlier if levels run out).

    Returns:
      List of lists of integers (selected indices for each partition).

    Notes:
      - Works for any N. If N is not divisible by 2**k, integer-rounded block
        boundaries introduce small discrepancies (often <= 1) in size/overlap.
      - Levels stop when number of blocks (2^(level+1)) exceeds N.
    """
    N = int(num_objects)
    if N <= 0:
        return []

    if num_partitions is not None:
        num_partitions = int(num_partitions)
        if num_partitions <= 0:
            return []

    partitions: List[List[int]] = []
    all_indices = list(range(N))

    level = 0
    while True:
        if num_partitions is not None and len(partitions) >= num_partitions:
            break

        num_blocks = 1 << (level + 1)  # 2^(level+1)
        if num_blocks > N:
            break

        selected = []
        # Block b covers [round(b*N/num_blocks), round((b+1)*N/num_blocks))
        # via integer rounding to nearest (ties up).
        for b in range(num_blocks):
            if b % 2 == 0:  # take even blocks: 0,2,4,...
                start = (b * N + num_blocks // 2) // num_blocks
                end = ((b + 1) * N + num_blocks // 2) // num_blocks
                if end > start:
                    selected.extend(range(start, end))

        selected = sorted(set(selected))
        sel_set = set(selected)
        complement = [i for i in all_indices if i not in sel_set]

        partitions.append(selected)
        if num_partitions is not None and len(partitions) >= num_partitions:
            break

        partitions.append(complement)

        level += 1

    return partitions


def partition_stats(partitions: List[List[int]]) -> (int, int):
    """
    Given a list of partitions (each a list of selected indices),
    returns:
      (min_partition_size, max_pairwise_overlap)

    Also prints those values.

    Notes:
      - Treats each partition as a set (duplicates ignored).
      - Overlap is |Pi ∩ Pj| for i < j.
    """
    if not partitions:
        print("min_size=0, max_overlap=0 (no partitions)")
        return 0, 0

    sets = [set(p) for p in partitions]
    sizes = [len(s) for s in sets]
    min_size = min(sizes)

    max_overlap = 0
    m = len(sets)
    for i in range(m):
        si = sets[i]
        for j in range(i + 1, m):
            ov = len(si & sets[j])
            if ov > max_overlap:
                max_overlap = ov

    print(f"min_size={min_size}, max_overlap={max_overlap}")
    return min_size, max_overlap


if __name__ == "__main__":
    logging.info("Hello World.")
