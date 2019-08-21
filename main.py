#! /usr/bin/env python3
# -*-Coding: utf-8 -*-

from src.read_dataset import TrainDataset, TestDataset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.separate_composit import composit_output
from src.image_processing import calc_psnr
from skimage.measure import compare_ssim
from itertools import product
from src.utils import confirm_make_folder
from src.read_dir_imags import ImgInDirAsY
from torch.nn.functional import relu
from torch.autograd import Variable
import time

SCALE = 2

# Prepare dataset
TRAIN_PATH = './dataset/Train'
TEST_DIR = os.path.join('dataset', 'Test')
TEST_DATASET_NAMES = ['Set5', 'Set14', 'BSD100']

train_transform = transforms.Compose([transforms.ToTensor()])
# train_transform = transforms.Compose([transforms.ToTensor(),
#                                       transforms.Normalize((0.5, ), (0.5, ))])


test_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = TrainDataset(dataset_path=TRAIN_PATH,
                             transform=train_transform)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=32, shuffle=True)

test_loaders = []
for test_dataset_name in TEST_DATASET_NAMES:
    test_dataset = TestDataset(os.path.join(TEST_DIR, test_dataset_name), transform=test_transform)
    test_loaders.append(torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1))

# Prepare result folder
MODEL_DIR = 'model'
confirm_make_folder(MODEL_DIR)
RESULT_DIR = 'result'
confirm_make_folder(RESULT_DIR)


# Define network
class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 5))
        self.conv2 = nn.Conv2d(64, 32, (5, 5))
        self.conv3 = nn.Conv2d(32, SCALE ** 2, (3, 3))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        return self.conv3(x)


device = 'cuda'
net = CnnNet().to(device)

criterion = nn.MSELoss()
criterion = criterion.to(device)

lr_params = [{'params': net.conv1.parameters(), 'lr': 0.0001},
             {'params': net.conv2.parameters(), 'lr': 0.0001},
             {'params': net.conv3.parameters(), 'lr': 0.00001}]

optimizer = optim.SGD(params=lr_params, momentum=0.9, weight_decay=0.0005)

print('Show cnn setting', net, optimizer, sep='\n')

NUM_EPOCH = 1000000

# Prepaer training
df_all_columns_list = ['epoch', 'train_loss', 'val_loss'].extend(
    ['{}_{}'.format(x, y) for x, y in product(['psnr', 'ssim'], TEST_DATASET_NAMES)])


df_all = pd.DataFrame(index=[], columns=df_all_columns_list)


for epoch in range(1, NUM_EPOCH+1):
    train_loss = 0
    train_cnt = 0
    val_loss = 0
    val_cnt = 0
    # Training
    net.train()
    for cnn_input, cnn_label in train_loader:

        cnn_input = Variable(cnn_input).to(device)
        cnn_label = Variable(cnn_label).to(device)

        optimizer.zero_grad()
        cnn_output = net(cnn_input)

        loss = criterion(cnn_output, cnn_label)
        train_loss += loss.item()
        train_cnt += 1

        loss.backward()
        optimizer.step()
    ave_train_loss = train_loss / len(train_loader)

    # if epoch % 10 == 0:
    #     print('Epoch [{}/{}], train_loss: {train_loss:.6f}'.format(epoch,
    #                                                                NUM_EPOCH, train_loss=ave_train_loss))

    if epoch % 1000 != 0:
        continue

    # Evaluation
    torch.save(net.state_dict(), '{}/model_epoch_{}.ckpt'.format(MODEL_DIR, epoch))
    net.eval()
    with torch.no_grad():
        df_all_epoch = pd.Series(epoch, index=['epoch'])
        for test_loader, test_dataset_name in zip(test_loaders, TEST_DATASET_NAMES):
            confirm_make_folder(os.path.join(RESULT_DIR, str(epoch), test_dataset_name))

            test_file_names = ImgInDirAsY(os.path.join(
                TEST_DIR, test_dataset_name)).files_name()
            df_epoch = pd.DataFrame(index=[], columns=['name', 'psnr', 'ssim', 'time'])
            for (cnn_input, cnn_label), test_file_name in zip(test_loader, test_file_names):
                cnn_input = cnn_input.to(device)
                cnn_label = cnn_label.to(device)

                time_sta = time.perf_counter()

                cnn_output = net(cnn_input)

                time_end = time.perf_counter()
                tim = time_end - time_sta

                # validation loss
                loss = criterion(cnn_output, cnn_label)
                val_loss += loss.item()
                val_cnt += 1

                # convert mat image
                cnn_output = cnn_output.cpu().numpy() * 255
                cnn_output = cnn_output.astype(np.uint8)
                cnn_output = composit_output(cnn_output)

                cnn_label = cnn_label.cpu().numpy() * 255
                cnn_label = cnn_label.astype(np.uint8)
                cnn_label = composit_output(cnn_label)

                # psnr and ssim
                psnr_result = calc_psnr(cnn_label, cnn_output)
                ssim_result = compare_ssim(cnn_label, cnn_output)

                image_name = os.path.splitext(os.path.basename(test_file_name))[0]
                outim_name = './{0}/{1}/{2}/{3}_{1}.bmp'.format(
                    RESULT_DIR, epoch, test_dataset_name, image_name)
                cv2.imwrite(outim_name, cnn_output)

                df_epoch = df_epoch.append(
                    pd.Series(
                        [image_name, psnr_result, ssim_result, tim],
                        index=df_epoch.columns),
                    ignore_index=True)

            # calculation average
            psnr_average = df_epoch['psnr'].mean()
            ssim_average = df_epoch['ssim'].mean()
            tim_average = df_epoch['time'].mean()

            df_epoch = df_epoch.append(
                pd.Series(['average', psnr_average, ssim_average, time_average],
                          index=df_epoch.columns),
                ignore_index=True)

            # save df_epoch
            df_epoch.to_pickle(
                '{0}/{1}/{2}/df_epoch_{1}.pkl'.format(RESULT_DIR, epoch, test_dataset_name))
            df_epoch.to_csv('{0}/{1}/{2}/df_epoch_{1}.csv'.format(RESULT_DIR,
                                                                  epoch, test_dataset_name))

            # add df_epoch result
            df_all_epoch['psnr_{}'.format(test_dataset_name)] = psnr_average
            df_all_epoch['ssim_{}'.format(test_dataset_name)] = ssim_average
            df_all_epoch['time_{}'.format(test_dataset_name)] = time_average

        ave_val_loss = val_loss / val_cnt
    df_all = df_all.append(df_all_epoch, ignore_index=True)

    print('Epoch [{}/{}], train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f},psnr_Set5: {psnr_Set5: .2f}, psnr_Set14: {psnr_Set14: .2f}, psnr_BSD100: {psnr_BSD100: .2f} '
          .format(epoch,
                  NUM_EPOCH, train_loss=ave_train_loss, val_loss=ave_val_loss,
                  psnr_Set5=df_all_epoch['psnr_Set5'],
                  psnr_Set14=df_all_epoch['psnr_Set14'],
                  psnr_BSD100=df_all_epoch['psnr_BSD100']))

df_all.to_pickle('{}/df_all.pkl'.format(RESULT_DIR))
df_all.to_csv('{}/df_all.csv'.format(RESULT_DIR))

print(df_all.iloc[df_all['psnr_BSD100'].idxmax()])
