"""
The MIT License (MIT)

Copyright (c) 2019 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import logging
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import math

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def normalize(img, whitening=False, verbose=False):
    img = img.astype(np.float32)

    if np.abs(np.max(img) - np.min(img)) < 1e-10:
        img[:] = 0
        if verbose:
            logging.warning("Image is Constant.")
        return img

    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    if whitening:
        std_cap = 1.0 / np.sqrt((len(img)))

        adjusted_stddev = max(np.std(img), std_cap)
        img = (img - np.mean(img)) / adjusted_stddev

    return img


def show(*args, nrows=1, backend=None, title=None, cmap=None, **kwargs):
    if backend is not None:
        matplotlib.use(backend)

    num_images = len(args)

    ncols = int(np.ceil(num_images / nrows) + 0.01)

    fig, axes = plt.subplots(nrows, ncols, **kwargs)

    if title is not None:
        fig.suptitle(title)

    if num_images > 1:
        for ax, img in zip(axes.flatten(), args):
            ax.imshow(img, cmap=cmap)
    else:
        axes.imshow(args[0], cmap=cmap)

    plt.show()
    return fig


def plot(
    *images,
    bins=60,
    nrows=1,
    nhists=2,
    min_val=None,
    max_val=None,
    title="Voxel/Pixel Value Distribution",
    log_scale=False,
    **kwargs,
):
    """
    Plots a histogram of the voxel/pixel values in one or more images.

    Parameters:
    - *images: unpacked sequence of 2D or 3D numpy arrays representing the images.
    - bins: int, optional. Number of bins for the histogram. Default is 100.
    - nrows: int, optional. Number of rows for subplots when there are multiple. Default is 1.
    - nhists: int, optional. Maximum number of histograms per subplot. Default is 2.
    - min_val: float or int, optional. Minimum value to be included in the histogram. Default is None (no minimum).
    - max_val: float or int, optional. Maximum value to be included in the histogram. Default is None (no maximum).
    - title: str, optional. The title for the histogram plot. Default is "Voxel/Pixel Value Distribution".
    - log_scale: bool, optional. If True, sets the y-axis of the histogram to a logarithmic scale. Default is False.

    Returns:
    - None
    """

    num_images = len(images)
    total_subplots = math.ceil(num_images / nhists)
    ncols = math.ceil(total_subplots / nrows)

    sns.set_palette("pastel", n_colors=num_images)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)

    # Handle different configurations of axs for consistent indexing
    if nrows == 1 and ncols == 1:
        axs = [[axs]]
    elif nrows == 1:
        axs = [axs]
    elif ncols == 1:
        axs = [[ax] for ax in axs]

    current_subplot = 0
    for idx, image in enumerate(images):
        if idx % nhists == 0 and idx != 0:
            current_subplot += 1

        ax = axs[current_subplot // ncols][current_subplot % ncols]

        flattened_image = image.flatten()

        if min_val is not None:
            flattened_image = flattened_image[flattened_image >= min_val]
        if max_val is not None:
            flattened_image = flattened_image[flattened_image <= max_val]

        sns.histplot(
            flattened_image,
            bins=bins,
            kde=False,
            color=sns.color_palette("pastel")[idx % nhists],
            ax=ax,
            label=f"Image {idx + 1}",
        )

        ax.set_xlabel("Voxel/Pixel Value")
        ax.set_ylabel("Frequency")
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()


if __name__ == "__main__":
    logging.info("Hello World.")
