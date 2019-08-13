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

SCALE = 2
STRIDE_SIZE = 8
AROUND_SHAVE = 2
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
        return cnn_inputs, cnn_labels

    def __len__(self):
        return self.dataset.result_len()


class SeparateSmallImgs():
    def __init__(self, dataset_path):
        self.img_files = ImgInDirAsY(dataset_path)
        self.cnn_inputs = self.prepare_inputs()
        self.cnn_teaches = self.prepare_labels()

    def get_input(self, index):
        return self.cnn_inputs[index, :, :, :]

    def get_label(self, index):
        return self.cnn_teaches[index, :, :, :]

    def prepare_inputs(self):
        cnn_inputs_list = []
        for hr_img in self.img_files.read_files():
            hr_img = modcrop(hr_img, SCALE)
            _wid, _hei = hr_img.shape
            for i, (y, x) in enumerate(zip(range(0, _hei - INPUT_SIZE*SCALE, STRIDE_SIZE),
                                           range(0, _wid - INPUT_SIZE*SCALE, STRIDE_SIZE))):
                hr_patch = hr_img[x: x+INPUT_SIZE*SCALE, y: y+INPUT_SIZE*SCALE]
                lr_patch = cv2.resize(
                    hr_patch, dsize=None, fx=1/SCALE, fy=1/SCALE, interpolation=cv2.INTER_CUBIC)
                lr_patch = lr_patch[AROUND_SHAVE: -AROUND_SHAVE,
                                    AROUND_SHAVE: -AROUND_SHAVE]
                lr = np.zeros(
                    (1, INPUT_SIZE-AROUND_SHAVE*2, INPUT_SIZE-AROUND_SHAVE*2), dtype=np.double)
                lr[0, :, :] = lr_patch
                cnn_inputs_list.append(lr)
        cnn_inputs = np.array(cnn_inputs_list, dtype=np.uint8)
        return cnn_inputs

    def prepare_labels(self):
        cnn_labels_list = []
        for hr_img in self.img_files.read_files():
            hr_img = modcrop(hr_img, SCALE)
            _wid, _hei = hr_img.shape
            for i, (y, x) in enumerate(zip(range(0, _hei - INPUT_SIZE*SCALE, STRIDE_SIZE),
                                           range(0, _wid - INPUT_SIZE*SCALE, STRIDE_SIZE))):
                hr_patch = hr_img[x: x+INPUT_SIZE*SCALE, y: y+INPUT_SIZE*SCALE]
                hr_patch_separete = separate_label(
                    hr_patch, SCALE)[
                    :, SHAVE_SIZE + AROUND_SHAVE - 1: -(SHAVE_SIZE + AROUND_SHAVE + 1),
                    SHAVE_SIZE + AROUND_SHAVE - 1: -(SHAVE_SIZE + AROUND_SHAVE + 1)]
                hr = np.zeros((SCALE ** 2, LABEL_SIZE-AROUND_SHAVE*2,
                               LABEL_SIZE-AROUND_SHAVE*2), dtype=np.double)
                hr[:, :, :] = hr_patch_separete
                cnn_labels_list.append(hr)
        cnn_lanels = np.array(cnn_labels_list, dtype=np.uint8)
        return cnn_lanels

    def result_len(self):
        cnt = 0
        for hr_img in self.img_files.read_files():
            hr_img = modcrop(hr_img, SCALE)
            _wid, _hei = hr_img.shape
            for i, (y, x) in enumerate(zip(range(0, _hei - INPUT_SIZE*SCALE, STRIDE_SIZE),
                                           range(0, _wid - INPUT_SIZE*SCALE, STRIDE_SIZE))):
                cnt += 1
        return cnt


class TestDataset(Dataset):
    def __init__(self, dataset_path, trainsform=None, target_transform=None):
        self.images_files = ImgInDirAsY(dataset_path)

    def __getitem__(self, index):
        ref = self.images_files.read_file(index)
        ref = modcrop(ref, SCALE)
        input = cv2.resize(ref, dsize=None, fx=1/2, fy=1/2,
                           interpolation=cv2.INTER_CUBIC)
        ref = separate_label(ref, SCALE)[:, SHAVE_SIZE - 1: -(SHAVE_SIZE + 1),
                                         SHAVE_SIZE - 1: -(SHAVE_SIZE + 1)]

        return input.astype(np.uint8) / 255, ref.astype(np.uint8) / 255

    def __len__(self):
        return self.images_files.files_len()
