#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np


def confirm_make_folder(*args):
    for arg in args:
        if not os.path.isdir(arg):
            os.makedirs(arg)


def round_off(x, d=0):
    p = 10 ** d
    return float(math.floor((x * p) + math.copysign(0.5, x))) / p


def check_files_len(*args):
    try:
        if args == [args[0]] * len(args) if args else False:
            raise ValueError(
                'Error: Check that img_files amount is same')
    except ValueError as err:
        print(err)


def check_img_size(*args: np.ndarray):
    try:
        for arg in args:
            if not arg.shape == args[0].shape:
                raise ValueError('Error: Check the image size')
    except ValueError as err_file:
        print(err_file)
