#! /usr/bin/env python3
# -*-Coding: utf-8 -*-

from src.read_dataset import TrainDataset, TestDataset
import torch
import torch.nn as nn
import torchvison.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.separate_composit import composit_output
from src.image_processing import calc_psnr
from skimage.measure import ssim

SCALE = 2

# Prepare dataset
TRAIN_PATH = './dataset/Train'
TEST_DIR = os.path.join('dataset', 'Test')
TEST_DATASET_NAMES = ['Set5', 'Set14', 'BSD100']

train_transform = transforms.Compose([transforms.ToTesor,
                                      transforms.Normalize([0.5], [0.5])])

train_dataset = TrainDataset(dataset_path=TRAIN_PATH,
                             transforms=train_transform,
                             shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32)

test_loaders = []
for test_dataset_name in TEST_DATASET_NAMES:
    test_dataset = TestDataset(os.path.join(TEST_DIR, test_dataset_name))
    test_loaders.append(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1))

# Define network


class CnnNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2D(1, 64, (5, 5))
        self.conv2 = nn.Conv2D(64, 32, (5, 5))
        self.conv3 = nn.Conv2D(32, SCALE**2, (3, 3))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)


device = 'cuda'
net = CnnNet().to(device)

criterion = nn.MESLoss()

lr_params = [{net.conv1.parameters(): lr= 0.0001},
             {net.conv2.parameters(): lr= 0.0001},
             {net.conv3.parameters(): lr= 0.00001}]

optimizer = optim.SGD(, momentum=0.9, weight_decay=0.0005)

NUM_EPOCH = 100

# Prepaer training
df_all_columns_list =
df_all = pd.DataFrame(index=[], columns=[])


for epoch in range(1, NUM_EPOCH+1):
    train_loss = 0
    val_loss = 0
    # Training
    net.train()
    for cnn_input, cnn_label in train_loader:
        cnn_input = cnn_input.float().to(device)
        cnn_label = cnn_label.float().to(device)

        optimizer.zero_grad()
        cnn_output = net(cnn_input)

        loss = criterion(cnn_output, cnn_label)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
    ave_train_loss = train_loss / len(train_loader.dataset)
    # Evaluation
    if epoch % 10 == 0:
        torch.save(net.state_dict(), 'temp.ckpt')
        net.eval()
        with torch.no_grad():

            for test_loader, test_dataset_name in zip(test_loaders, TEST_DATASET_NAMES):

                for cnn_input, cnn_label in test_loader:
                    cnn_input = cnn_input.float().to(device)
                    cnn_label = cnn_label.float().to(device)

                    cnn_output = net(cnn_input)

                    # validation loss
                    loss = criterion(cnn_output, cnn_label)
                    val_loss += loss.item()

                    # psnr and ssim
                    cnn_output = cnn_output.cpu().numpy()
                    cnn_output = cnn_output.astype(np.uint8)
                    cnn_output = composit_output(cnn_output) * 255

                    cnn_label = cnn_label.cpu().numpy()
                    cnn_label = cnn_label.astype(np.uint8)
                    cnn_label = composit_output(cnn_label) * 255

                    psnr_result = calc_psnr(cnn_label, cnn_output)
                    ssim_result = ssim(cnn_label, cnn_output)

                ave_val_loss = val_loss / len(test_loader.dataset)
        print('Epoch [{}/{}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}'.format(epoch,
                                                                                             NUM_EPOCH, train_loss=ave_train_loss))
    else:
        print('Epoch [{}/{}], train_loss: {train_loss:.4f}'.format(epoch,
                                                                   NUM_EPOCH, train_loss=ave_train_loss))
