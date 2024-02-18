# Consult gan/explanation.md for more information on how this model works.

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def weights_init(m):
    '''
    Method to initialize weights for Generator and Discriminator networks
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, NGPU, IMG_SIZE, IMG_CHANNELS, Z_SIZE):
        '''
        Initializes the Generator transposed convolutional neural network. 
        IMG_SIZE is the feature map size and Z_SIZE is the size of the latent vector input.
        '''
        super(Generator, self).__init__()
        self.ngpu = NGPU
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(Z_SIZE, IMG_SIZE * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(IMG_SIZE * 8),
            nn.ReLU(True),
            # state size. ``(IMG_SIZE*8) x 4 x 4``
            nn.ConvTranspose2d(IMG_SIZE * 8, IMG_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMG_SIZE * 4),
            nn.ReLU(True),
            # state size. ``(IMG_SIZE*4) x 8 x 8``
            nn.ConvTranspose2d(IMG_SIZE * 4, IMG_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMG_SIZE * 2),
            nn.ReLU(True),
            # state size. ``(IMG_SIZE*2) x 16 x 16``
            nn.ConvTranspose2d(IMG_SIZE * 2, IMG_SIZE, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMG_SIZE),
            nn.ReLU(True),
            # state size. ``(IMG_SIZE) x 32 x 32``
            nn.ConvTranspose2d(IMG_SIZE, IMG_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(3) x 64 x 64``
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, NGPU, IMG_SIZE, IMG_CHANNELS):
        '''
        Initializes the Discriminator convolutional neural network. 
        IMG_SIZE is the feature map size.
        '''
        super(Discriminator, self).__init__()
        self.ngpu = NGPU
        self.main = nn.Sequential(
            # input is ``(3) x 64 x 64``
            nn.Conv2d(IMG_CHANNELS, IMG_SIZE, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(IMG_SIZE) x 32 x 32``
            nn.Conv2d(IMG_SIZE, IMG_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMG_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(IMG_SIZE*2) x 16 x 16``
            nn.Conv2d(IMG_SIZE * 2, IMG_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMG_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(IMG_SIZE*4) x 8 x 8``
            nn.Conv2d(IMG_SIZE * 4, IMG_SIZE * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMG_SIZE * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(IMG_SIZE*8) x 4 x 4``
            nn.Conv2d(IMG_SIZE * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)