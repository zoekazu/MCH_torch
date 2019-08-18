#!/usr/bin/env python3
# -*- Coding: utf-8 -*-

import numpy as np
from itertools import product
import math


def separate_label(img_in: np.ndarray, scale: int):
    _hei, _wid = img_in.shape

    img_out = np.zeros((scale ** 2, _hei // 2, _wid // 2), dtype=np.uint8)

    for cnt, (_y, _x) in enumerate(product(range(scale), range(scale))):
        img_out[cnt, :, :] = img_in[_y:_hei:scale, _x:_wid:scale]
    return img_out


def composit_output(img_in: np.ndarray):
    _, _ch, _wid, _hei = img_in.shape
    _scale = int(math.sqrt(_ch))

    img_out = np.zeros((_wid * 2, _hei * 2), dtype=np.uint8)

    for cnt, (_y, _x) in enumerate(product(range(_scale), range(_scale))):
        img_out[_y:_wid*2:_scale, _x:_hei*2:_scale] = img_in[0, cnt, :, :]
    return img_out
