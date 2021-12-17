""" This module contains the AI models that will be used in
the model_training and image_analysis modules for training """

from image_transform import image_transform, semantic_segmentation, binary_transformation
from PIL import Image
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

# This model class encodes data in the model horizon
class Encoder(nn.Module):
    def __init__(self, input_size=150528, hidden_size=388, num_layers=1, p=0.5):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, sequence):
        sequence = self.dropout(sequence)
        output, (hidden, cell) = self.lstm(sequence)
        return hidden

# This model class decodes encoding from sequence to make prediction
class Decoder(nn.Module):
    def __init__(self, lstm_hidden_size=388, linear_hidden1_size=776, linear_hidden2_size=28, linear_hidden3_size=12, output_size=1):
        super(Decoder, self).__init__()
        self.hidden1 = nn.Linear(lstm_hidden_size, linear_hidden1_size)
        self.hidden2 = nn.Linear(linear_hidden1_size, linear_hidden2_size)
        self.hidden3 = nn.Linear(linear_hidden2_size, linear_hidden3_size)
        self.predict = nn.Linear(linear_hidden3_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, encoding):
        hidden_layer1 = self.hidden1(encoding)
        hidden_layer2 = self.hidden2(hidden_layer1)
        hidden_layer3 = self.hidden3(hidden_layer2)
        predictions = self.predict(hidden_layer3)
        output = self.activation(predictions)
        return output

# This model class combines the encoder-decoder to create the overall architecture
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sequence):    
        # First encode sequence of images then flatten encoded data to size (batch_size, hidden_size) 
        encoding = self.encoder(sequence)
        encoding = encoding.reshape(-1, encoding.shape[2])
        prediction = self.decoder(encoding)
        return prediction

# Initialize weights in all submodels of the AI (LSTM and Linear layers)
def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
        m.bias.data.fill_(0)


""" Instantiate pre-trained segmentation model as a class object """
fcn_model = models.segmentation.fcn_resnet101(pretrained=True)

""" Instantiate scoring model as a class object """
encoder_model = Encoder()
decoder_model = Decoder()
lstm_model = Seq2Seq(encoder_model, decoder_model)
lstm_model.apply(init_weights)