import torch as pt
from collections import OrderedDict
import numpy as np

import torch.nn as nn


class SequenceToImageCNN(nn.Module):
    """ An encode-decoder CNN that takes in a sinogram (D x N) and outputs an image (N x N).
    Each row in the sinogram is a projection (1xN), and the number of projections is D.
    The encoder is an LSTM that takes in the sinogram, and the decoder is a CNN that takes in the state of the LSTM after
    all projections have been fed in.
    """
    def __init__(self, input_shape, output_shape, hidden_size=128, num_layers=1):
        super(SequenceToImageCNN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.encoder = nn.LSTM(input_size=input_shape[2], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Now, the encoder will take in a sinogram, and output a hidden state containing the information of the sinogram
        # The decoder will take in the hidden state, and output a vector with prod(output_shape) elements
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, np.prod(output_shape)),
            nn.Sigmoid()
        )

        
    def forward(self, s):
        # Pad the sinogram to have input_shape[1] rows
        s = s.squeeze(0)
        s = pt.nn.functional.pad(s, (0, 0, 0, self.input_shape[1] - s.shape[0]))
        s = s.reshape((1, self.input_shape[1], self.input_shape[2]))
        # Pass the sinogram through the LSTM, and get the hidden state
        _, (h_n, c_n) = self.encoder(s)
        # Pass the hidden state through the decoder
        dec = self.decoder(h_n)
        dec = dec.reshape(self.output_shape)
        return dec

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
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

class UNetWithSinogramInput(UNet):
    """ This class implements a UNet that takes in a sinogram as input, and outputs an image.
    Since they can be different shapes, we pad the smaller one with zeros.
    """
    def __init__(self, input_shape, output_shape, in_channels=1, out_channels=1, init_features=32):
        # Input shape is the shape of the sinogram, output shape is the shape of the image
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.pad_to_shape = (max(input_shape[0], output_shape[0]), output_shape[1])
        super(UNetWithSinogramInput, self).__init__(in_channels, out_channels, init_features)
        
    def forward(self, s):
        # Pad the sinogram to the shape of the image
        #print(f"UnetwithSinogramInput input shape: {s.shape}")
        # Pad to self.pad_to_shape
        s = s.squeeze(0)
        s = pt.nn.functional.pad(s, (0, self.pad_to_shape[1] - s.shape[1], 0, self.pad_to_shape[0] - s.shape[0]))
        #print(f"UnetwithSinogramInput padded input shape: {s.shape}")
        out = super().forward(s.unsqueeze(0).unsqueeze(0))
        # Only output output_shape
        #print(f"UnetwithSinogramInput output shape: {out.shape}")
        out = out[:, :, 0:self.output_shape[0], 0:self.output_shape[1]]
        out = out.squeeze(0).squeeze(0)
        #print(f"UnetwithSinogramInput cropped output shape: {out.shape}")
        return out