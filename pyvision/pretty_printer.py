"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import logging
import math

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


# Class for colors
class Colors:
    RED = '\033[31;1m'
    GREEN = '\033[32;1m'
    YELLOW = '\033[33;1m'
    BLUE = '\033[34;1m'
    MAGENTA = '\033[35;1m'
    CYAN = '\033[36;1m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


# Colored value output if colorized flag is activated.
def get_color_entry(val, args=None):
    if not isinstance(val, float) or math.isnan(val):
        color = Colors.ENDC
        return color
    if (val < .25):
        color = Colors.RED
    elif (val < .50):
        color = Colors.YELLOW
    elif (val < .70):
        color = Colors.BLUE
    elif (val < .85):
        color = Colors.CYAN
    else:
        color = Colors.GREEN

    return color

NEW_TABLE_LINE_MARKER = -111


def pretty_print_table(names, value_list, header_names,
                       step=None, color_results=True):

    value_list_t = zip(*value_list)

    ind = 8 * " "

    table_head = ind + "{:^18} |" + " {:^12} |" * len(header_names)

    table_line = ind + '-' * (20 + 15 * len(header_names))
    table_entry = ind + "{:<18} |" + " {:>12} |" * len(header_names)

    logging.info('')
    logging.info("Evaluation Results:")
    logging.info('')
    logging.info('')

    logging.info(table_head.format('Score', *header_names))
    logging.info(table_line)

    for name, vals in zip(names, value_list_t):
        if not vals[0] == NEW_TABLE_LINE_MARKER:
            if color_results:
                logging.info(table_entry.format(name, *_color_result(vals)))
            else:
                logging.info(table_entry.format(name, *vals))
        else:
            logging.info(table_line)

    logging.info(table_line)
    logging.info('')
    logging.info('')


def _color_result(vals, val_format="{:>12.2f}"):
    cvals = []
    for val in vals:
        cval = get_color_entry(val) + val_format.format(100 * val) \
            + Colors.ENDC
        cvals.append(cval)
    return tuple(cvals)

if __name__ == '__main__':
    logging.info("Hello World.")
