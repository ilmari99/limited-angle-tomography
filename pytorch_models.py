import json
import math
import os
import torch as pt
import torch
import itertools
from torch import Tensor, nn
from collections import OrderedDict
import numpy as np
from torch_radon import Radon
from utils import FBPRadon
from regularization import extract_patches_2d_pt, reconstruct_from_patches_2d_pt




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
    
    
class SinogramToPatchesConnected(torch.nn.Module):
    def __init__(self, image_size, patch_size, stride, angles, a=0):
        """ A model that uses a connected architecture to predict
        an image from a sinogram.
        The model is only locally connected, so that only the relevant
        part of the sinogram is used to predict the patch at a certain location.
        Args:   
            image_size: The size of the image
            patch_size: The size of a patch, i.e. the local receptive field
            stride: The stride of the patches. Overlapping patches are averaged.
            angles: The angles (rad) used in the Radon transform
        """
        super(SinogramToPatchesConnected, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.angles = angles
        #print(f"Image size: {image_size}, Patch size: {patch_size}, Stride: {stride}")
        
        self.radont = FBPRadon(image_size, angles, a=a)
        
        # Get the base_matrices and the corresponding sinograms
        patch_start_pixels = self.get_patch_starts(image_size, patch_size, stride=stride)
        #print(f"Number of patch_starts: {len(patch_start_pixels)}")
        
        #num_patches = extract_patches_2d_pt(torch.zeros((image_size, image_size), device='cuda'), patch_size, stride=stride)
        #print(f"Number of patches: {num_patches.shape}", flush=True)
        
        # Calculate which patches are actually inside the circle List[bool]
        patch_is_inside_circle = []
        for start in patch_start_pixels:
            i, j = start
            dist_from_center_to_start = np.sqrt((i - image_size // 2) ** 2 + (j - image_size // 2) ** 2)
            dist_from_center_to_end = np.sqrt((i + patch_size - image_size // 2) ** 2 + (j + patch_size - image_size // 2) ** 2)
            if dist_from_center_to_start <= image_size // 2 and dist_from_center_to_end <= image_size // 2:
                patch_is_inside_circle.append(True)
            else:
                patch_is_inside_circle.append(False)
        print(f"Num inside circle: {sum(patch_is_inside_circle)}, Num outside circle: {len(patch_is_inside_circle) - sum(patch_is_inside_circle)}")
        # Find every img x img mask, where only the patch is 1
        patch_masks = []
        for patch_idx, start in enumerate(patch_start_pixels):
            i, j = start
            mask = np.zeros((image_size, image_size))
            if patch_is_inside_circle[patch_idx]:
                mask[i:i+patch_size, j:j+patch_size] = 1
            patch_masks.append(mask)
        
        patch_masks = np.array(patch_masks)
        base_sinograms = self.get_base_sinograms(patch_masks, angles)
        #self.base_sinograms.to("cpu")
        #self.patch_masks = torch.tensor(self.patch_masks, dtype=torch.float32, device='cpu')
        # TODO: The values at base_sinograms could actually be used as sort of attention weights
        
        # The sinogram masks are the sinograms, but every != 0 value is set to 1
        masks = []
        for sinogram in base_sinograms:
            mask = torch.where(sinogram > 1e-6, 1, 0)
            masks.append(mask)
        avg_num_of_ones_in_mask = sum([torch.sum(mask).item() for mask in masks]) / len(masks)
        print(f"Avg num of ones in mask: {avg_num_of_ones_in_mask}")
        self.sinogram_masks = torch.stack(masks).to("cpu").to(torch.float16)
        print(f"Masks: {self.sinogram_masks.shape}")
        
        # We use a single model to predict each patch, based on it's masked sinogram
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(len(self.angles) * self.image_size, self.patch_size * self.patch_size),
        )
        self.model.to('cuda')
    
    def forward(self, sinogram):
        # Get the masked sinograms
        sinogram = sinogram.to('cpu').to(torch.float16)
        # Elementwise multiplication
        masked_sinograms = sinogram * self.sinogram_masks
        #print(f"Masked sinograms shape: {masked_sinograms.shape}")
        # As input, give a masked sinogram, and predict a patch_size x patch_size patch
        # npatches x 1 x num_angles x image_size
        masked_sinograms = masked_sinograms.unsqueeze(1)
        patches = []
        # Do in batchs of 64
        batch_size = 32
        for i in range(0, masked_sinograms.shape[0], batch_size):
            n = batch_size if i + batch_size < masked_sinograms.shape[0] else masked_sinograms.shape[0] - i
            batch = masked_sinograms[i:i+n]
            batch = batch.to('cuda').float()
            patch_batch = self.model(batch)
            patch_batch = torch.reshape(patch_batch, (n, self.patch_size, self.patch_size))
            patches.append(patch_batch.cpu().to(torch.float16))
        patches = torch.cat(patches)
        patches = patches.squeeze()
        #print(f"Patches shape: {patches.shape}")
        #patches = patches.squeeze()
        # Reconstruction from patches
        y_hat = reconstruct_from_patches_2d_pt(patches, (self.image_size, self.image_size), stride=self.stride, device='cpu')
        y_hat = y_hat.to('cuda').float()
        y_hat = torch.sigmoid(y_hat)
        return y_hat
    
    @staticmethod
    def get_patch_starts(image_size, patch_size, stride=1):
        """ Return the starting pixel of each patch.
        """
        patch_starts = []
        for i in range(0, image_size, stride):
            for j in range(0, image_size, stride):
                if i + patch_size > image_size or j + patch_size > image_size:
                    continue
                patch_starts.append((i, j))
        return patch_starts
    
    @staticmethod
    def get_base_sinograms(base_matrices, angles, a=0) -> list:
        """ Get the sinograms of the base matrices.
        """
        rt = FBPRadon(base_matrices.shape[1], angles, a)
        base_sinograms = []
        for mat in base_matrices:
            mat = torch.tensor(mat, dtype=torch.float32, device='cuda')
            sinogram = rt.forward(mat)
            sinogram = sinogram.cpu()
            base_sinograms.append(sinogram)
        return base_sinograms

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
    def __init__(self, input_shape, init_features=32, overwrite_cache=False, load_weights="", init_channels=2):
        super().__init__()
        assert init_features >= 4, "init_features must be at least 8"
        assert init_features % 4 == 0, "init_features must be divisible by 4"
        #assert input_shape[1] == 512, "Input shape must have 512 columns"


        def blocks(dim, n):
            return [Block(dim) for _ in range(n)]
        
        
        encoder_kernel1 = (4, 4)
        encoder_stride1 = (4, 4)
        encoder_padding1 = (1, 1)
        encoder_kernel2 = (3, 4)
        encoder_stride2 = (3, 4)
        encoder_padding2 = (1, 1)
        encoder_kernel3 = (3, 4)
        encoder_stride3 = (2, 4)
        encoder_padding3 = (1, 0)
        
        # Calculate the output shape of the encoder
        # The input shape is (B, 1, N, D)
        # [(W−K+2P)/S]+1
        N = input_shape[0]
        D = input_shape[1]
        
        in_cache = self.check_cache(N, D, 8)
        if in_cache and not overwrite_cache:
            (encoder_kernel1,
             encoder_stride1,
             encoder_padding1,
             encoder_kernel2,
             encoder_stride2,
             encoder_padding2,
             encoder_kernel3,
             encoder_stride3,
             encoder_padding3) = in_cache
        else:
            (encoder_kernel1,
             encoder_stride1,
             encoder_padding1,
             encoder_kernel2,
             encoder_stride2,
             encoder_padding2,
             encoder_kernel3,
             encoder_stride3,
             encoder_padding3) = self.find_kernel_stride_padding(N, D, 8)
        
        
        #print(f"Encoder kernel1: {encoder_kernel1}")
        #print(f"Encoder stride1: {encoder_stride1}")
        #print(f"Encoder padding1: {encoder_padding1}")
        #print(f"Encoder kernel2: {encoder_kernel2}")
        #print(f"Encoder stride2: {encoder_stride2}")
        #print(f"Encoder padding2: {encoder_padding2}")
        #print(f"Encoder kernel3: {encoder_kernel3}")
        #print(f"Encoder stride3: {encoder_stride3}")
        
        # Output from encoder should be (B, 4*init_features, 8, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(init_channels, init_features, encoder_kernel1, encoder_stride1, encoder_padding1),
            #nn.Conv2d(1, init_features, (4, 4), (2, 4), (1, 1)),
            nn.BatchNorm2d(init_features),
            *blocks(init_features, n=1),
            nn.BatchNorm2d(init_features),
            nn.Conv2d(init_features, init_features*2, encoder_kernel2, encoder_stride2, encoder_padding2),
            #nn.Conv2d(init_features, init_features*2, (3, 4), (2, 4), (1, 1)),
            *blocks(init_features*2, n=3),
            nn.BatchNorm2d(init_features*2),
            nn.Conv2d(init_features*2, init_features*4, encoder_kernel3, encoder_stride3, encoder_padding3),
            # If input is 60 x 512 -> (B, 4*init_features, 3, 8)
        )
        
        
        self.decoder = nn.Sequential(
            *blocks(init_features*4, n=6),
            nn.BatchNorm2d(init_features*4),
            nn.ConvTranspose2d(init_features*4, init_features*2, 2, 2, 0),
            
            *blocks(init_features*2, n=3),
            nn.ConvTranspose2d(init_features*2, init_features, 2, 2, 0),
            
            *blocks(init_features, n=3),
            nn.ConvTranspose2d(init_features, init_features//4, 2, 2, 0),
            
            *blocks(init_features//4, n=1),
            
            nn.ConvTranspose2d(init_features//4, init_features//8, 2, 2, 0),
            *blocks(init_features//8, n=1),
            nn.ConvTranspose2d(init_features//8, init_features//8, 2, 2, 0),
            *blocks(init_features//8, n=1),
            
            nn.ConvTranspose2d(init_features//8, init_features//8, 2, 2, 0),
            *blocks(init_features//8, n=1),
            nn.Conv2d(init_features//8, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        
        if load_weights:
            try:
                self.load_state_dict(pt.load(load_weights))
            except Exception as e:
                print("Could not load weights: ", e)
        
    def check_cache(self, N, D, output_shape):
        if not os.path.exists("htc_model_params_cache.json"):
            return None
        with open("htc_model_params_cache.json", "r") as f:
            content = f.read()
            if content:
                content = json.loads(content)
                #print(f"Content: {content}")
                # Convert keys to tuples
                content = {eval(k):v for k,v in content.items()}
                #print(f"Content: {content}")
                if (N, D, output_shape) in content:
                    data = content[(N, D, output_shape)]
                    return tuple(data.values())
        return None
        
    def find_kernel_stride_padding(self, N, D, output_shape):
        grid = {
            "nk1":list(range(1, 6)),
            "ns1":list(range(1, 6)),
            "np1":list(range(0, 2)),
            "dk1":list(range(1,6)),
            "ds1":list(range(1, 6)),
            "dp1":list(range(0,2)),
            "nk2":list(range(1,6)),
            "ns2":list(range(1, 6)),
            "np2":list(range(0,2)),
            "dk2":list(range(1,6)),
            "ds2":list(range(1, 6)),
            "dp2":list(range(0,2)),
            "nk3":list(range(1,6)),
            "ns3":list(range(1, 6)),
            "np3":list(range(0,2)),
            "dk3":list(range(1,6)),
            "ds3":list(range(1, 6)),
            "dp3":list(range(0,2)),
        }
        
        # Loop through all possible combinations of kernel sizes, strides, and paddings
        for params in itertools.product(*grid.values()):
            nk1, ns1, np1, dk1, ds1, dp1, nk2, ns2, np2, dk2, ds2, dp2, nk3, ns3, np3, dk3, ds3, dp3 = params
            n_output_shape = self.calc_encoder_output_shape(N, nk1, ns1, np1, nk2, ns2, np2, nk3, ns3, np3)
            d_output_shape = self.calc_encoder_output_shape(D, dk1, ds1, dp1, dk2, ds2, dp2, dk3, ds3, dp3)
            if n_output_shape == output_shape and d_output_shape == output_shape:
                encoder_kernel1 = (nk1, dk1)
                encoder_stride1 = (ns1, ds1)
                encoder_padding1 = (np1, dp1)
                encoder_kernel2 = (nk2, dk2)
                encoder_stride2 = (ns2, ds2)
                encoder_padding2 = (np2, dp2)
                encoder_kernel3 = (nk3, dk3)
                encoder_stride3 = (ns3, ds3)
                encoder_padding3 = (np3, dp3)
                break
        
        # The content is dictionary, where keys are (N, D, output_shape) tuples,
        # and values are dictionaries with the kernel, stride, and padding values
        with open("htc_model_params_cache.json", "r") as f:
            content = f.read()
            if not content:
                content = "{}"
            content = json.loads(content)
            #print(f"Content: {content}")
        # cache
        with open("htc_model_params_cache.json", "w") as f:
        
            entry = {f"({N}, {D}, {output_shape})": {
                "encoder_kernel1":encoder_kernel1,
                "encoder_stride1":encoder_stride1,
                "encoder_padding1":encoder_padding1,
                "encoder_kernel2":encoder_kernel2,
                "encoder_stride2":encoder_stride2,
                "encoder_padding2":encoder_padding2,
                "encoder_kernel3":encoder_kernel3,
                "encoder_stride3":encoder_stride3,
                "encoder_padding3":encoder_padding3
            }}
            
            # Update content and write it back to the file
            content.update(entry)
            print(f"Writing content: {content}")
            f.write(json.dumps(content))
        return encoder_kernel1, encoder_stride1, encoder_padding1, encoder_kernel2, encoder_stride2, encoder_padding2, encoder_kernel3, encoder_stride3, encoder_padding3
        
    def calc_encoder_output_shape(self, d0, k1, s1, p1, k2, s2, p2, k3, s3, p3):
        #(((((d_0 - k_1 + 2p_1)/s_1 + 1) - k_2 + 2*p_2)/s_2 + 1) - k_3 +2*p_3) / s_3 + 1
        if (d0 - k1 + 2*p1) % s1 != 0:
            return 0
        if ((d0 - k1 + 2*p1) / s1 + 1 - k2 + 2*p2) % s2 != 0:
            return 0
        if ((((d0 - k1 + 2*p1)//s1 + 1 - k2 + 2*p2)//s2 + 1 - k3 + 2*p3) % s3) != 0:
            return 0
        return ((((d0 - k1 + 2*p1)//s1 + 1 - k2 + 2*p2)//s2 + 1 - k3 + 2*p3)//s3 + 1)

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