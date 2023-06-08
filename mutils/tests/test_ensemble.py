"""
The MIT License (MIT)

Copyright (c) 2023 Marvin Teichmann
Email: marvin.teichmann@googlemail.com

The above author notice shall be included in all copies or
substantial portions of the Software.

This file is written in Python 3.8 and tested under Linux.
"""

import pytest
from mutils.ensemble import (
    sliding_window_partition,
    generate_feature_partitions,
    create_partitions_check_overlap,
)

import logging
import sys
import math

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def test_sliding_window_partition():
    assert sliding_window_partition(list(range(5)), 3, 2) == [
        [0, 1, 2],
        [2, 3, 4],
    ]

    assert sliding_window_partition(list(range(6)), 3, 2) == [
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 0],
    ]


def test_generate_feature_partitions():
    groups = generate_feature_partitions(16, 4, 2)
    assert all(len(group) == 4 for group in groups)
    max_overlap = 4 // 2
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert len(set(groups[i]).intersection(groups[j])) <= max_overlap


def test_create_partitions_check_overlap(capsys):
    groups = create_partitions_check_overlap(16, 4, 2)
    captured = capsys.readouterr()
    assert "Generated " in captured.out
    assert len(groups) > 0


@pytest.mark.parametrize(
    "num_objects,group_size,overlap_ratio",
    [(16, 4, 2), (16, 2, 2), (256, 64, 2), (256, 32, 2), (512, 32, 2)],
)
def test_generate_feature_partitions_cases(
    num_objects, group_size, overlap_ratio
):
    groups = generate_feature_partitions(
        num_objects, group_size, overlap_ratio
    )

    assert all(len(group) == group_size for group in groups)

    max_overlap = group_size // overlap_ratio
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert len(set(groups[i]).intersection(groups[j])) <= max_overlap


if __name__ == "__main__":
    test_sliding_window_partition()
    test_generate_feature_partitions()

    # Define the test cases
    test_cases = [
        (16, 4, 2),
        (16, 2, 2),
        (256, 64, 2),
        (256, 32, 2),
        (512, 32, 2),
    ]

    # Iterate over the test cases and call the test function for each one
    for case in test_cases:
        test_generate_feature_partitions_cases(*case)

    pytest.main(["-v", __file__])
