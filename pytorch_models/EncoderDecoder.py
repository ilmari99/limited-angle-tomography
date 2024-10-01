import itertools
from typing import Iterator
import sympy
import torch
import math

from torch.nn.parameter import Parameter
from torch.nn import Conv2d

from utils import FBPRadon
from .utils import find_convolution_parameters, find_convolution_parameters_grid_search

class Block(torch.nn.Module):
    def __init__(self, dim, kernel_size=5, expansion=2):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv = Conv2d(dim, dim, kernel_size, 1, padding)
        self.norm = torch.nn.BatchNorm2d(dim)
        hidden_dim = int(expansion * dim)
        self.ln1 = Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = torch.nn.GELU()
        self.ln2 = Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ln1(out)
        out = self.act(out)
        out = self.ln2(out)
        out += x
        return out
    
def blocks(dim, n):
    return [Block(dim) for _ in range(n)]

class Decoder(torch.nn.Module):
    def __init__(self, latent_image_side_len, output_shape_side_len, filters = [32, 32, 32,1]):
        super(Decoder, self).__init__()
        self.encoder_output_side_len = latent_image_side_len
        if filters[-1] != 1:
            filters.append(1)
        self.filters = filters
        self.output_shape_side_len = output_shape_side_len
        self.num_conv_layers = len(filters) - 1
        parameters = find_convolution_parameters_grid_search(self.num_conv_layers,
                                                 self.encoder_output_side_len,
                                                 self.output_shape_side_len,
                                                 transposed_convolutions=True,
                                                 )

        self.kernel_sizes, self.paddings, self.strides = parameters
        if any([x is None for x in parameters]):
            raise ValueError("Could not find convolution parameters")
        self.decoder = self.create_decoder()
        #print(f"Decoder: {self.decoder}")
        
    def create_decoder(self):
        layers = []
        filters = self.filters
        for i in range(self.num_conv_layers):
            if i == 0:
                layers.extend(blocks(filters[i], 6))
                layers.append(torch.nn.BatchNorm2d(filters[i]))
            else:
                layers.extend(blocks(filters[i], 2))
            layers.append(torch.nn.ConvTranspose2d(filters[i],
                                                       filters[i+1],
                                                       kernel_size=self.kernel_sizes[i],
                                                       stride=self.strides[i],
                                                       padding=self.paddings[i]))
            if i == self.num_conv_layers - 1:
                break
            
        decoder = torch.nn.Sequential(*layers)
        return decoder

    def forward(self, x):
        # Firstly, we reshape the input to a square
        x = x.reshape((1, self.filters[0], self.encoder_output_side_len, self.encoder_output_side_len))
        # Then we pass it through the decoder
        x = self.decoder(x)
        # reshape it to the output shape
        x = x.reshape(-1, self.output_shape_side_len, self.output_shape_side_len)
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self, input_shape, angles, latent_image_side_len, filters = [32, 64, 128, 32]):
        """ A model that takes in a sinogram that is masked, so only the relevant region is visible.
        """
        super(Encoder, self).__init__()
        self.encoder_output_side_len = latent_image_side_len
        if len(input_shape) == 3:
            assert input_shape[0] == 1
            input_shape = input_shape[1:]
        
        self.input_shape = input_shape
        self.angles = angles
        if filters[0] != 1:
            filters.insert(0, 1)
        self.filters = filters
        self.num_layers = len(filters)-1
        
        h_params = find_convolution_parameters_grid_search(self.num_layers,
                                                            self.input_shape[0],
                                                            self.encoder_output_side_len,
                                                            transposed_convolutions = False,
                                                            grid_update={"kernel_size" : [3,5,7,9],
                                                                         "padding" : [0,1,2],
                                                                         "stride" : [1,2,4,6]
                                                                         })
        if any([x is None for x in h_params]):
            raise ValueError("Could not find convolution parameters for height")
        w_params = find_convolution_parameters_grid_search(self.num_layers,
                                                            self.input_shape[1],
                                                            self.encoder_output_side_len,
                                                            transposed_convolutions = False,
                                                            grid_update={"kernel_size" : [3,5,7,9,12],
                                                                         "padding" : [0,1,2],
                                                                         "stride" : [1,2,4,6,9]
                                                                         })
        if any([x is None for x in w_params]):
            raise ValueError("Could not find convolution parameters")
        h_kernels, h_paddings, h_strides = h_params
        w_kernels, w_paddings, w_strides = w_params
        
        self.kernel_sizes = [(h_kernels[i], w_kernels[i]) for i in range(self.num_layers)]
        self.paddings = [(h_paddings[i], w_paddings[i]) for i in range(self.num_layers)]
        self.strides = [(h_strides[i], w_strides[i]) for i in range(self.num_layers)]
        
        # Encoder-decoder architecture
        self.encoder = self.create_encoder()
        #print(f"Encoder: {self.encoder}")
        
    def create_encoder(self):
        layers = []
        filters = self.filters
        for i in range(self.num_layers):
            layers.append(torch.nn.Conv2d(filters[i],
                                              filters[i+1],
                                                kernel_size=self.kernel_sizes[i],
                                                stride=self.strides[i],
                                                padding=self.paddings[i]))
            if i == self.num_layers - 1:
                break
            if i == 0:
                layers.append(torch.nn.BatchNorm2d(filters[i+1]))
            #layers.append(torch.nn.ReLU())
            layers.extend(blocks(filters[i+1], 2))
            layers.append(torch.nn.BatchNorm2d(filters[i+1]))
            
        encoder = torch.nn.Sequential(*layers)
        return encoder
        
    def _create_encoder_with_gavg(self):
        layers = []
        filters = self.filters
        for i in range(len(filters) - 1):
            layers.append(torch.nn.Conv2d(filters[i],
                                              filters[i+1],
                                                kernel_size=3,
                                                stride=1,
                                                padding=1))
            # Don't add ReLU to the last layer
            if i == len(filters) - 2:
                break
            layers.append(torch.nn.ReLU())
        # Global average pooling to get the latent representation
        layers.append(torch.nn.AdaptiveAvgPool2d(1))
        encoder = torch.nn.Sequential(*layers)
        return encoder
    
    
    def forward(self, x):
        # Reshape if necessary
        if len(x.shape) == 2:
            x = x.reshape((1, 1, *x.shape))
        x = self.encoder(x)
        return x

class EncoderDecoder(torch.nn.Module):
    def __init__(self, input_shape, angles, latent_image_side_len, output_side_len = 512, encoder_filters = [32, 64], decoder_filters = [32, 64]):
        """ A model that takes in a sinogram that is masked, so only the relevant region is visible.
        """
        super(EncoderDecoder, self).__init__()
    
        self.input_shape = input_shape
        self.angles = angles
        self.output_side_len = output_side_len
        self.latent_image_side_len = latent_image_side_len
        
        self.encoder = Encoder(input_shape, angles, latent_image_side_len, encoder_filters)
        self.decoder = Decoder(self.latent_image_side_len, output_side_len, decoder_filters)
        
    def get_encoder_output_shape(self, input_shape):
        with torch.no_grad():
            x = torch.randn(input_shape)
            x = self.encoder(x)
        x = x.squeeze()
        # The output size must be a single square number
        assert math.sqrt(x.shape[0]).is_integer()
        return x.shape
    
    def forward(self, x):
        # Reshape if necessary
        if len(x.shape) == 2:
            x = x.reshape((1, 1, *x.shape))
        if len(x.shape) == 3:
            x = x.reshape((1, *x.shape))
        x = self.encoder(x)
        x = self.decoder(x)
        #x = torch.sigmoid(x)
        return x