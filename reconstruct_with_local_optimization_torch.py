
from collections import OrderedDict
from AbsorptionMatrices import Circle

import torch as pt
from torch_radon import Radon

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from reconstruct import filter_sinogram, reconstruct_outer_shape
from utils import reconstruct_error
from tqdm import tqdm

RADON = Radon(128, np.linspace(0,np.pi,180, endpoint=False), clip_to_circle=True)
        
def filter_sinogram(sino, a = 0.1):
    """filter projections. Normally a ramp filter multiplied by a window function is used in filtered
    backprojection. The filter function here can be adjusted by a single parameter 'a' to either approximate
    a pure ramp filter (a ~ 0)  or one that is multiplied by a sinc window with increasing cutoff frequency (a ~ 1).
    Credit goes to Wakas Aqram. 
    inputs: sino - [n x m] torch tensor where n is the number of projections and m is the number of angles used.
    outputs: filtSino - [n x m] filtered sinogram tensor
    
    Reference: https://github.com/csheaff/filt-back-proj
    """
    sino = sino.cpu().detach()
    sino = pt.squeeze(sino)
    sino = pt.transpose(sino, 0, 1)
    print(f"Sinogram shape: {sino.shape}")
    
    projLen, numAngles = sino.shape
    step = 2*np.pi/projLen
    w = pt.arange(-np.pi, np.pi, step)
    if len(w) < projLen:
        w = pt.cat([w, w[-1]+step]) #depending on image size, it might be that len(w) =  
                                        #projLen - 1. Another element is added to w in this case
    rn1 = abs(2/a*pt.sin(a*w/2))  #approximation of ramp filter abs(w) with a funciton abs(sin(w))
    rn2 = pt.sin(a*w/2)/(a*w/2)   #sinc window with 'a' modifying the cutoff freqs
    r = rn1*(rn2)**2                 #modulation of ramp filter with sinc window
    
    filt = pt.fft.fftshift(r)
    filtSino = pt.zeros((projLen, numAngles))
    
    for i in range(numAngles):
        projfft = pt.fft.fft(sino[:,i])
        filtProj = projfft*filt
        filtSino[:,i] = pt.real(pt.fft.ifft(filtProj))

    return filtSino.T

def decompose_and_reconstruct(matrix, radon_t) -> np.ndarray:
    """
    Decompose a matrix into a sinogram, and then reconstruct it back.
    """
    # Matrix must be a cuda GPU tensor
    sinogram = radon_t.forward(matrix)
    reconstruction = radon_t.backprojection(sinogram)
    
    # Apply a filter to the reconstruction
    #reconstruction = filter_sinogram(reconstruction)
    
    # Scale to 0-1
    reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())#
    return reconstruction
    
class NNet(nn.Module):
    """ Simple dense neural network """
    def __init__(self, input_shape):
        super(NNet, self).__init__()
        self.w1 = nn.Parameter(pt.randn(input_shape[0]*input_shape[1], 128))
        self.b1 = nn.Parameter(pt.randn(128))
        self.w2 = nn.Parameter(pt.randn(128, 128))
        self.b2 = nn.Parameter(pt.randn(128))
        self.w3 = nn.Parameter(pt.randn(128, input_shape[0]*input_shape[1]))
        self.b3 = nn.Parameter(pt.randn(input_shape[0]*input_shape[1]))
        
        self.input_shape = input_shape
        self.to('cuda')

    def forward(self, x):
        x = x.view(-1, self.input_shape[0]*self.input_shape[1])
        x = pt.relu(pt.matmul(x, self.w1) + self.b1)
        x = pt.relu(pt.matmul(x, self.w2) + self.b2)
        x = pt.relu(pt.matmul(x, self.w3) + self.b3)
        return x.view(self.input_shape)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, image_size=128):
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
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.upconv2(bottleneck)
        dec1 = pt.cat((dec2, enc2), dim=1)
        dec1 = self.decoder2(dec1)
        dec0 = self.upconv1(dec1)
        dec0 = pt.cat((dec0, enc1), dim=1)
        dec0 = self.decoder1(dec0)
        return pt.sigmoid(self.conv(dec0))

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
    
class CustomModel(nn.Module):
    def __init__(self, input_shape, angles, outer_mask = None):
        super(CustomModel, self).__init__()
        if outer_mask is None:
            outer_mask = pt.ones(input_shape, device='cuda')
        self.angles = angles
        self.angles_rad = np.deg2rad(angles)
        self.radon_t = Radon(input_shape[0], self.angles_rad, clip_to_circle=True)
        # Load keras model from 'blurry_to_real_model2.keras'
        self.u_net = UNet(in_channels=1, out_channels=1, init_features=32, image_size=input_shape[0])
        self.decompose_and_reconstruct = lambda x: decompose_and_reconstruct(x, self.radon_t)
        
    def forward(self, x):
        x = x.float().cuda()
        y_hat = self.u_net(x)
        # Multiply elemnetwise with outer mask
        y_hat = y_hat * outer_mask
        y_hat_prime = self.decompose_and_reconstruct(y_hat)
        return y_hat, y_hat_prime
    
circle = Circle(63)
circle.make_holes(10, n_missing_pixels=0.5, inplace=True)
angles = np.arange(0,180,1)
y = circle.matrix
# Pad to 128x128
y = np.pad(y, ((0,1),(0,1)))
reconstruction_errors = []

measurements,distances_from_front,distances_from_back = circle.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
thicknesses = np.full(measurements.shape, measurements.shape[1])
thicknesses = thicknesses - distances_from_front - distances_from_back
outer_mask = reconstruct_outer_shape(angles,distances_from_front,distances_from_back,zero_threshold=0.1)
outer_mask = pt.tensor(outer_mask, dtype=pt.float32, device='cuda')
# Pad to 128x128
outer_mask = pt.nn.functional.pad(outer_mask, (0,1,0,1))

model = CustomModel((128,128), angles, outer_mask = outer_mask)
model.to('cuda')


optimizer = optim.Adam(model.parameters(), lr=0.1)
y_ = pt.tensor(y, dtype=pt.float32, device='cuda', requires_grad=True)
y_prime = decompose_and_reconstruct(y_, model.radon_t)
y_prime_cpu = y_prime.cpu().detach().numpy()
#y_prime_pt = y_prime.clone().detach().requires_grad_(True)
print(f"Shapes: y: {y.shape}, y_prime: {y_prime.shape}")

y_hat_fig, y_hat_ax = plt.subplots(1,3, figsize=(10,5))
# Plot the true y, and the predicted y
y_hat_ax[0].matshow(y)
y_hat_ax[1].matshow(y_prime_cpu)
plt.show(block=False)
criterion = nn.MSELoss()
while True:
    # Calculate y_hat = decompose_and_reconstruct(y_prime)
    # Calculate y_hat_prime = model(y_hat)
    # minimize ||y_prime - y_hat_prime||^2
    
    y_hat, y_hat_prime = model(y_prime)
    
    # minimize ||y_prime - y_hat_prime||^2
    print(f"Shapes: y_hat: {y_hat.shape}, y_hat_prime: {y_hat_prime.shape}")
    print(f"Shapes: y_prime: {y_prime.shape}")
    
    # Calculate the loss
    loss = criterion(y_hat_prime, y_prime.unsqueeze(0))
    print(f"Loss: {loss.item()}")
    
    # Update the model
    # Retain grads
    loss.backward(retain_graph=True)

    
    optimizer.step()
    optimizer.zero_grad()
    
    y_hat_prime = y_hat_prime.squeeze()
    y_hat = y_hat.squeeze()
    
    
    # Squeeze and detach
    y_hat = y_hat.cpu().detach().numpy()
    y_hat_prime = y_hat_prime.cpu().detach().numpy()
    reconstruction_errors.append(reconstruct_error(y, y_hat))
    
    print(f"Reconstruction error for True image: {reconstruction_errors[-1]}")
    print(f"Error between y_prime and y_hat_prime: {loss}")
    
    # Update the figure
    y_hat_ax[2].matshow(y_hat)
    y_hat_fig.canvas.draw()
    y_hat_fig.canvas.flush_events()
    plt.pause(0.1)
    if reconstruction_errors[-1] < 0.001:
        break

