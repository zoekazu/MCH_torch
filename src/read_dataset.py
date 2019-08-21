#! /usr/bin/env python3

from torch.utils.data import Dataset
import os
import sys
import glob
from src.read_dir_imags import ImgInDirAsY
from src.separate_composit import separate_label
from itertools import product
import cv2
import numpy as np
from src.image_processing import modcrop
import torchvision.transforms as transforms
from src.image_processing import image_shave

SCALE = 2
STRIDE_SIZE = 8
INPUT_SIZE = 30
LABEL_SIZE = 20
SHAVE_SIZE = abs(INPUT_SIZE - LABEL_SIZE) // 2


class TrainDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset = SeparateSmallImgs(dataset_path)
        self.transform = transform

    def __getitem__(self, index):
        cnn_inputs = self.dataset.get_input(index)
        cnn_labels = self.dataset.get_label(index)

        if self.transform:
            return self.transform(cnn_inputs), self.transform(cnn_labels)

    def __len__(self):
        return self.dataset.result_len()


class SeparateSmallImgs():
    def __init__(self, dataset_path):
        self.img_files = ImgInDirAsY(dataset_path)
        self.cnn_inputs, self.cnn_teaches, self.cnt = self.prepare_dataset()

    def get_input(self, index):
        return self.cnn_inputs[index, :, :, :]

    def get_label(self, index):
        return self.cnn_teaches[index, :, :, :]

    def result_len(self):
        return self.cnt

    def prepare_dataset(self):
        cnn_inputs_list = []
        cnn_labels_list = []
        cnt = 0

        for hr_img in self.img_files.read_files():
            hr_img = modcrop(hr_img, SCALE)
            _hei, _wid = hr_img.shape

            for y, x in zip(range(0, _hei - INPUT_SIZE*SCALE, STRIDE_SIZE),
                            range(0, _wid - INPUT_SIZE*SCALE, STRIDE_SIZE)):
                hr = hr_img[y: y+INPUT_SIZE*SCALE, x: x+INPUT_SIZE*SCALE]
                hr = separate_label(hr, SCALE)
                hr = image_shave(hr, SHAVE_SIZE, SHAVE_SIZE)

                cnn_labels_list.append(hr)

                lr = cv2.resize(hr_patch, dsize=None, fx=1/SCALE,
                                fy=1/SCALE, interpolation=cv2.INTER_CUBIC)
                lr = lr[:, :, np.newaxis]
                cnn_inputs_list.append(lr)

                cnt += 1
        cnn_inputs = np.array(cnn_inputs_list, dtype=np.uint8)
        cnn_labels = np.array(cnn_labels_list, dtype=np.uint8)

        return cnn_inputs, cnn_labels, cnt


class TestDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.images_files = ImgInDirAsY(dataset_path)
        self.transform = transform

    def __getitem__(self, index):
        cnn_label = self.images_files.read_file(index)
        cnn_label = modcrop(cnn_label, SCALE)

        cnn_input = cv2.resize(cnn_label, dsize=None, fx=1/2, fy=1/2,
                               interpolation=cv2.INTER_CUBIC)
        cnn_input = cnn_input[:, :, np.newaxis]

        cnn_label = separate_label(cnn_label, SCALE)
        cnn_label = image_shave(cnn_label, SHAVE_SIZE, SHAVE_SIZE)

        cnn_input = cnn_input.astype(np.uint8)
        cnn_label = cnn_label.astype(np.uint8)

        if self.transform:
            return self.transform(cnn_input), self.transform(cnn_label)

    def __len__(self):
        return self.images_files.files_len()
