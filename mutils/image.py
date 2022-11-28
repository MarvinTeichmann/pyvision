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


def show(*args, nrows=1, backend=None, title=None, **kwargs):

    if backend is not None:
        matplotlib.use(backend)

    num_images = len(args)

    ncols = int(np.ceil(num_images / nrows) + 0.01)

    fig, axes = plt.subplots(nrows, ncols, **kwargs)

    if title is not None:
        fig.suptitle(title)

    if num_images > 1:
        for ax, img in zip(axes.flatten(), args):
            ax.imshow(img)
    else:
        axes.imshow(args[0])

    plt.show()
    return fig


plot = show


if __name__ == "__main__":
    logging.info("Hello World.")
