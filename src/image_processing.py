#! /usr/bin/env python3
# -*- Coding: utf-8 -*-

import numpy as np
import cv2
import math


def modcrop(img, scale):
    if img.ndim == 3:
        img_size = np.array(img.shape[0:2])
        img_size = img_size - np.mod(img_size, scale)
        out_img = img[0:img_size[0], 0: img_size[1], :]
    else:
        img_size = np.array(img.shape[0:2])
        img_size = img_size - np.mod(img_size, scale)
        out_img = img[0:img_size[0], 0: img_size[1]]

    return out_img


def psnr_calc(ref, img):
    mse = np.mean((ref - img) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
