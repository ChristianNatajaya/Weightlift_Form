""" This module contains functions that train the AI models
which is executed in the image_analysis module """

import visualization_models as AI
from image_transform import image_transform, semantic_segmentation, binary_transformation
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import cv2
import os

def train_segmentation_model(train_data, dev_data, model, lr=0.01, momentum=0.9, nesterov=False, n_epochs=5):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)

    # First train with training set then validate with test set
    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(train_data, model.train(), optimizer)
        val_loss = run_epoch(dev_data, model.eval(), optimizer)
        print("-------------\nEpoch {}:\n".format(epoch))
        print('Train loss: {:.6f} | Validation loss: {:.6f}'.format(train_loss, val_loss))

def run_epoch(data, model, optimizer):
    losses_class = []
    is_training = model.training

    # Iterate through batches and compute losses (Currently each batch has 1 image only)
    for batch in tqdm(data):
        x, y = batch['x'].float(), batch['y'].float()
        prediction = model(x)['out'].float()
        prediction = torch.square(prediction)
        max_value = torch.max(prediction)
        prediction = torch.div(prediction, max_value)

        loss = F.binary_cross_entropy(prediction, y)
        losses_class.append(loss.data.item())
        
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    avg_loss = np.mean(losses_class)
    return avg_loss

def train_scoring_model(train_data, dev_data, model, lr=0.001, momentum=0.9, nesterov=False, n_epochs=50):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)

    # First train with training set then validate with test set
    for epoch in range(1, n_epochs + 1):
        train_loss = run_scoring_epoch(train_data, model.train(), optimizer)
        val_loss = run_scoring_epoch(dev_data, model.eval(), optimizer)
        print("-------------\nEpoch {}:\n".format(epoch))
        print('Train loss: {:.6f} | Validation loss: {:.6f}'.format(train_loss, val_loss))

def run_scoring_epoch(data, model, optimizer):
    losses_class = []
    is_training = model.training

    # Iterate through different video sequences
    for sequence in tqdm(data):
        # Flatten (15,3,224,224) to (batch_size, seq_len, input_size) = (1,15,3*224*224)
        lstm_input = sequence['x'].float()
        lstm_input = lstm_input.view(1, lstm_input.size()[0], -1)
        # print(lstm_input.size())

        prediction = model(lstm_input).float()
        target = torch.Tensor(sequence['y']).unsqueeze(0)
        loss = F.mse_loss(prediction, target)
        print(prediction, target, loss)
        losses_class.append(loss.data.item())
        
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25)
            optimizer.step()

    avg_loss = np.mean(losses_class)
    return avg_loss