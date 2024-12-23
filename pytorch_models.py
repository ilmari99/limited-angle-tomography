import json
import math
import os
import torch as pt
import torch
import itertools
from torch import Tensor, nn
from collections import OrderedDict
import numpy as np
from utils import (FBPRadon,
                   extract_patches_2d_pt,
                   reconstruct_from_patches_2d_pt,
                   )

class FixedCNN(nn.Module):
    """ This model takes in an image with just random noise,
    passes the noise through a series of convolutional layers,
    and outputs a final image of the desired shape.   
    """
    def __init__(self, output_shape, num_layers=3):
        super(FixedCNN, self).__init__()
        self.output_shape = (output_shape, output_shape)
        self.num_layers = num_layers
        self.conv_layers = self.create_conv_layers()
        self.conv_layers = self.conv_layers.train()
    
    def parameters(self, recurse: bool = True):
        return self.conv_layers.parameters(recurse)
        
    def create_conv_layers(self):
        """ Create a series of convolutional layers
        """
        layers = []
        self.init_features = 64
        for i in range(self.num_layers):
            in_channels = 1 if i == 0 else self.init_features
            out_channels = self.init_features
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
        # The output is a single channel image
        layers.append(nn.Conv2d(self.init_features, 1, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv_layers(x)
        return out

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, load_weights=""):
        """ UNet model for image reconstruction
        
        Args:
            in_channels: int
                Number of input channels
            out_channels: int
                Number of output channels
            init_features: int
                Number of features in the first layer
            image_size: int
        """
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 2, features * 4, name="bottleneck")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        
        if load_weights:
            self.load_state_dict(pt.load(load_weights))
            

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.upconv2(bottleneck)
        dec1 = pt.cat((dec2, enc2), dim=1)
        dec1 = self.decoder2(dec1)
        dec0 = self.upconv1(dec1)
        dec0 = pt.cat((dec0, enc1), dim=1)
        dec0 = self.decoder1(dec0)
        #return self.conv(dec0)
        out_soft = pt.sigmoid(self.conv(dec0))
        return out_soft

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )