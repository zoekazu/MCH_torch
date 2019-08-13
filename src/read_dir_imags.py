#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import glob
import cv2


class ImgsInDir():
    def __init__(self, file_dir, *, file_type='bmp'):
        self.file_dir = file_dir
        self.file_type = file_type

        dir_name = os.path.join(self.file_dir, '*.{}'.format(self.file_type))
        self.img_files = glob.glob(dir_name)

        try:
            if self.img_files is None:
                raise ValueError('Reading directory is empty')
        except ValueError as err_dir:
            print(err_dir)

    def read_file(self, file_num):
        return cv2.imread(self.img_files[file_num], cv2.IMREAD_COLOR)

    def read_files(self):
        for i in range(self.files_len()):
            yield self.read_file(i)

    def files_len(self): return len(self.img_files)

    def file_name(self, file_num): return self.img_files[file_num]

    def files_name(self):
        for i in range(self.files_len()):
            yield self.file_name(i)


class ImgsInDirAsGray(ImgsInDir):
    def __init__(self, file_dir, *, file_type='bmp', threshold_level=127):
        super().__init__(file_dir)
        self._threshould_level = threshold_level

    def read_file(self, file_num):  # Over ride
        return cv2.imread(self.img_files[file_num], cv2.IMREAD_GRAYSCALE)


class ImgInDirAsY(ImgsInDir):
    def __init__(self, file_dir, *, file_type='bmp'):
        super().__init__(file_dir)

    def read_file(self, file_num):  # over ride
        _img = cv2.imread(self.img_files[file_num], cv2.IMREAD_ANYCOLOR)
        if _img.ndim == 3:
            return cv2.cvtColor(_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        else:
            return _img


class ImgsInDirAsBool(ImgsInDir):
    def __init__(self, file_dir, *, file_type='bmp', threshold_level=127, bool_switch=False):
        super().__init__(file_dir)
        self._threshould_level = threshold_level
        self._bool_switch = bool_switch

    def read_file(self, file_num):  # Over ride
        _img = cv2.imread(self.img_files[file_num], cv2.IMREAD_GRAYSCALE)

        _, _img = cv2.threshold(
            _img, self._threshould_level, 255, cv2.THRESH_BINARY)
        _img = _img.astype(bool)

        if self._bool_switch:
            _out_img = np.logical_not(_img)
        else:
            _out_img = _img

        return _out_img
