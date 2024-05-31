import math
import torch as pt
import torch
from torch import Tensor, nn
from collections import OrderedDict
import numpy as np
from torch_radon import Radon
from utils import FBPRadon
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig


class Block(nn.Module):
    def __init__(self, dim, kernel_size=5, expansion=2):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, padding)
        self.norm = nn.BatchNorm2d(dim)
        hidden_dim = int(expansion * dim)
        self.ln1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = nn.GELU()
        self.ln2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ln1(out)
        out = self.act(out)
        out = self.ln2(out)
        out += x
        return out


class HTCModel(nn.Module):
    def __init__(self, input_shape, init_features=32):
        super().__init__()
        assert init_features >= 8, "init_features must be at least 8"
        assert init_features % 4 == 0, "init_features must be divisible by 4"
        assert input_shape[1] == 512, "Input shape must have 512 columns"
        

        def blocks(dim, n):
            return [Block(dim) for _ in range(n)]
        
        # Output from encoder should be (B, 4*init_features, 8, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, init_features, (4, 4), (4, 4), (1, 1)),
            nn.BatchNorm2d(init_features),
            *blocks(init_features, n=1),
            nn.BatchNorm2d(init_features),
            nn.Conv2d(init_features, init_features*2, (3, 4), (3, 4), (1, 1)),
            *blocks(init_features*2, n=3),
            nn.BatchNorm2d(init_features*2),
            nn.Conv2d(init_features*2, init_features*4, (3, 4), (2, 4), (1, 0)),    # If input is 60 x 512 -> (B, 4*init_features, 3, 8)
        )
        
        
        self.decoder = nn.Sequential(
            *blocks(init_features*4, n=6),
            nn.BatchNorm2d(init_features*4),
            nn.ConvTranspose2d(init_features*4, init_features*2, 2, 2, 0),
            *blocks(init_features*2, n=3),
            nn.ConvTranspose2d(init_features*2, init_features, 2, 2, 0),
            *blocks(init_features, n=3),
            nn.ConvTranspose2d(init_features, init_features//2, 2, 2, 0),
            *blocks(init_features//2, n=1),
            nn.ConvTranspose2d(init_features//2, init_features//4, 2, 2, 0),
            *blocks(init_features//4, n=1),
            nn.ConvTranspose2d(init_features//4, init_features//4, 2, 2, 0),
            *blocks(init_features//4, n=1),
            
            nn.ConvTranspose2d(init_features//4, init_features//4, 2, 2, 0),
            *blocks(init_features//4, n=1),
            nn.Conv2d(init_features//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

class SinogramCompletionBase(nn.Module):
    """ Given a part of a sinogram, predict the rest of the sinogram.
    """
    def __init__(self,dim, input_sinogram_angles, output_sinogram_angles, output_yhat=False, device='cuda'):
        """
        Pytorch model that takes in a sinogram (N x D) and outputs a sinogram (M x D).
        Essentially the model only predicts (M - N x D) elements. The input sinogram is the first N
        projections of the output sinogram.

        Args:
        N : number of projections
        D : number of angles
            input_sinogram_shape: tuple
                Shape of the input sinogram, without the batch dimension (N x D)
            output_sinogram_shape: tuple
                Shape of the output sinogram, without the batch dimension (M x D)
            step_size: int
                How many degrees the object was rotated between each projection.
        """
        super(SinogramCompletionBase, self).__init__()
        self.input_sinogram_angles = input_sinogram_angles
        self.output_sinogram_angles = output_sinogram_angles
        self.output_yhat = output_yhat
        self.device = device

        self.dim = dim
        self.radon_t_backward = FBPRadon(resolution=self.dim, angles=self.output_sinogram_angles, device=self.device)
        self.nn = self.create_nn()
        
    def create_nn(self):
        raise NotImplementedError("Subclasses must implement this method")



class LSTMSinogram(SinogramCompletionBase):
    """ An LSTM model that takes in a sinogram (N x D) and outputs a sinogram (M x D)
    """
    def __init__(self, dim, input_sinogram_angles, output_sinogram_angles, output_yhat=False, device='cuda'):
        self.input_shape = (len(input_sinogram_angles), dim)
        self.output_shape = (len(output_sinogram_angles), dim)
        self.to_predict = (len(output_sinogram_angles) - len(input_sinogram_angles), dim)
        super(LSTMSinogram, self).__init__(dim, input_sinogram_angles, output_sinogram_angles, output_yhat, device)
        
    def create_nn(self):
        return SinogramCompletionTransformer(self.dim, self.input_sinogram_angles, self.output_sinogram_angles, output_yhat=self.output_yhat, device=self.device)
        #return SequenceToImageCNN((1,*self.input_shape), (1,*self.to_predict), hidden_size=self.input_shape[1], num_layers=self.to_predict[0])

        
    def forward(self, s):
        print(f"Input shape: {s.shape}")
        s = s.reshape((1, *self.input_shape))
        pred = self.nn(s)
        print(f"Predicted shape: {pred.shape}")
        pred = pred.reshape(self.to_predict[0], self.to_predict[1])
        # Concatenate the input sinogram with the predicted sinogram
        out = pt.cat((s.squeeze(0).squeeze(0), pred), dim=0)
        print(f"Output size: {out.shape}")
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SinogramCompletionTransformer(SinogramCompletionBase):
    """ A transformer model that takes in a sinogram (N x D) and outputs a completion to the sinogram (M x D).
    The input is passed through len(output_sinogram_angles) layers of the transformer, and the output is the last layer.
    """
    def __init__(self, dim, input_sinogram_angles, output_sinogram_angles, output_yhat=False, device='cuda'):
        super(SinogramCompletionTransformer, self).__init__(dim, input_sinogram_angles, output_sinogram_angles, output_yhat, device)
        self.input_shape = (len(input_sinogram_angles), dim)
        self.output_shape = (len(output_sinogram_angles), dim)
        self.to_predict = (len(output_sinogram_angles) - len(input_sinogram_angles), dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(self.input_shape[0], self.to_predict[0])
        
    def create_nn(self):
        return nn.Transformer(d_model=self.dim,
                              nhead=16,
                              num_encoder_layers=12,
                              num_decoder_layers=12,
                              dim_feedforward=2048,
                              dropout=0.1,
                              activation='relu')
    
    def forward(self, src):
        # src: (N, D)
        src = src.reshape((1, *self.input_shape))
        #src = self.pos_encoder(src)
        print(f"Input shape: {src.shape}")
        output = self.nn(src, src)  # (1, N, D)
        print(f"Transformer output shape: {output.shape}")
        output = self.fc1(output.squeeze(0))  # (N, D)
        print(f"FC1 output shape: {output.shape}")
        output = self.fc2(output.transpose(0, 1))  # (D, M)
        print(f"FC2 output shape: {output.shape}")
        return output.transpose(0, 1)  # (M, D)
        
                                                        
                                                        
        

class SequenceToImageCNN(nn.Module):
    """ An encoder-decoder CNN that takes in a sinogram (N x D) and outputs an image (N x N).
    Each row in the sinogram is a projection (1xN), and the number of projections is D.
    The encoder is an LSTM that takes in the sinogram, and the decoder is a CNN that takes in the state of the LSTM after
    all projections have been fed in.
    """
    def __init__(self, input_shape, output_shape, hidden_size=128, num_layers=1):
        super(SequenceToImageCNN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.encoder = nn.LSTM(input_size=input_shape[2], hidden_size=hidden_size, num_layers=1, batch_first=True,bidirectional=False)
        
        # Now, the encoder will take in a sinogram, and output a hidden state containing the information of the sinogram
        # The decoder will take in the hidden state, and output a vector with prod(output_shape) elements
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, np.prod(output_shape)),
            #nn.Sigmoid()
        )

        
    def forward(self, s):
        print(f"Input shape: {s.shape}")
        # Pad the sinogram to have input_shape[1] rows
        s = s.squeeze(0)
        #s = pt.nn.functional.pad(s, (0, 0, 0, self.input_shape[1] - s.shape[0]))
        #s = s.reshape((1, self.input_shape[1], self.input_shape[2]))
        s = s.reshape(*self.input_shape)
        print(f"Reshaped input shape: {s.shape}")
        # Pass the sinogram through the LSTM, and get the hidden state
        _, (h_n, c_n) = self.encoder(s)
        print(f"Hidden state shape: {h_n.shape}")
        
        # Pass the hidden state through the decoder
        dec = self.decoder(h_n)
        print(f"Decoded shape: {dec.shape}")
        dec = dec.reshape(self.output_shape)
        print(f"Reshaped decoded shape: {dec.shape}")
        return dec
    
class UNet2(nn.Module):
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
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                in_channels=in_channels, out_channels=out_channels, init_features=init_features)
        if load_weights:
            model.load_state_dict(pt.load(load_weights))
            
        super(UNet2, self).__init__()
        
        self.model = model
        
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        out = self.model(x)
        #print(f"Output shape: {out.shape}")
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