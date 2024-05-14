
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

def filter_sinogram(sino, a = 0.1):
    """filter projections. Normally a ramp filter multiplied by a window function is used in filtered
    backprojection. The filter function here can be adjusted by a single parameter 'a' to either approximate
    a pure ramp filter (a ~ 0)  or one that is multiplied by a sinc window with increasing cutoff frequency (a ~ 1).
    Credit goes to Wakas Aqram. 
    inputs: sino - [n x m] torch tensor where n is the number of projections and m is the number of angles used.
    outputs: filtSino - [n x m] filtered sinogram tensor
    
    Reference: https://github.com/csheaff/filt-back-proj
    """
    sino = pt.squeeze(sino)
    sino = sino.T
    #print(f"Sinogram shape: {sino.shape}")
    
    projLen, numAngles = sino.shape
    step = 2*np.pi/projLen
    w = pt.arange(-np.pi, np.pi, step, device='cuda')
    if len(w) < projLen:
        w = pt.cat([w, w[-1]+step]) #depending on image size, it might be that len(w) =  
                                        #projLen - 1. Another element is added to w in this case
    #print(w)
    rn1 = abs(2/a*pt.sin(a*w/2))  #approximation of ramp filter abs(w) with a funciton abs(sin(w))
    rn2 = pt.sin(a*w/2)/(a*w/2)   #sinc window with 'a' modifying the cutoff freqs
    r = rn1*(rn2)**2                 #modulation of ramp filter with sinc window
    filt = pt.fft.fftshift(r)
    # The first element in filt is Nan
    filt[0] = 0
    filtSino = pt.zeros((projLen, numAngles), device='cuda')
    
    for i in range(numAngles):
        projfft = pt.fft.fft(sino[:,i])
        filtProj = projfft*filt
        #print(f"Filt proj shape: {filtProj.shape}")
        #print(filtProj)
        ifft_filtProj = pt.fft.ifft(filtProj)
        
        filtSino[:,i] = pt.real(ifft_filtProj)

    return filtSino.T

def decompose_and_reconstruct(matrix, radon_t) -> pt.Tensor:
    """
    Decompose a matrix into a sinogram, and then reconstruct it back.
    """
    # Matrix must be a cuda GPU tensor
    sinogram = radon_t.forward(matrix)
    sinogram = filter_sinogram(sinogram)
    
    reconstruction = radon_t.backprojection(sinogram)
    
    # Scale
    reconstruction = (reconstruction - pt.min(reconstruction)) / (pt.max(reconstruction) - pt.min(reconstruction))
    
    return reconstruction

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
        #return self.conv(dec0)
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
        # Load weights from keras model
        self.u_net.load_state_dict(pt.load("model_gpu.pt"))
        self.u_net.to('cuda')
        #print(self.u_net)
            
        self.decompose_and_reconstruct = lambda x: decompose_and_reconstruct(x, self.radon_t)
        
    def forward(self, x):
        x = x.float().cuda()
        y_hat = self.u_net(x)
        # Multiply elemnetwise with outer mask
        y_hat = y_hat * outer_mask
        # Scale to [0,1]
        y_hat = (y_hat - pt.min(y_hat)) / (pt.max(y_hat) - pt.min(y_hat))
        y_hat_prime = self.decompose_and_reconstruct(y_hat)
        return y_hat, y_hat_prime
    
circle = Circle(63)
nholes = 10
hole_volatility = 0.5
n_missing_pixels = 0.4
hole_ratio_limit = 10
at_angle = np.random.randint(0,90)
circle.make_holes(nholes,
                  n_missing_pixels,
                  hole_volatility,
                  hole_ratio_limit,
                  at_angle)
#angles = np.random.choice(np.arange(0,180), 20, replace=False)
angles = np.arange(0,30,1)
y = circle.matrix
# Pad to 128x128
y = np.pad(y, ((0,1),(0,1)))

measurements,distances_from_front,distances_from_back = circle.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
thicknesses = np.full(measurements.shape, measurements.shape[1])
thicknesses = thicknesses - distances_from_front - distances_from_back
outer_mask = reconstruct_outer_shape(angles,distances_from_front,distances_from_back,zero_threshold=0.1)
outer_mask = pt.tensor(outer_mask, dtype=pt.float32, device='cuda')
# Pad to 128x128
outer_mask = pt.nn.functional.pad(outer_mask, (0,1,0,1))

model = CustomModel((128,128), angles, outer_mask = outer_mask)
model.to('cuda')


optimizer = optim.Adam(model.parameters(), lr=0.001)
y_ = pt.tensor(y, dtype=pt.float32, device='cuda', requires_grad=True)
y_prime = decompose_and_reconstruct(y_, model.radon_t)
y_prime = y_prime * outer_mask

# Scale to [0,1]
y_prime = (y_prime - pt.min(y_prime)) / (pt.max(y_prime) - pt.min(y_prime))

y_prime_cpu = y_prime.squeeze().cpu().detach().numpy()
#y_prime_pt = y_prime.clone().detach().requires_grad_(True)
print(f"Shapes: y: {y.shape}, y_prime: {y_prime.shape}")

y_hat_fig, y_hat_ax = plt.subplots(1,3, figsize=(10,5))
# Plot the true y, and the predicted y
y_hat_ax[0].matshow(y)
y_hat_ax[1].matshow(y_prime_cpu)
plt.show(block=False)
criterion = nn.MSELoss(reduction="mean")
iteration_number = 0

loss_fig, loss_ax = plt.subplots(1,2, figsize=(10,5))
loss_ax[0].set_title("Reconstruction error")
loss_ax[0].set_xlabel("Iteration number")
loss_ax[0].set_ylabel("Reconstruction error")
loss_ax[1].set_title("Loss")
loss_ax[1].set_xlabel("Iteration number")
loss_ax[1].set_ylabel("Loss")


reconstruction_errors = []
losses = []

while True:
    # Calculate y_hat = decompose_and_reconstruct(y_prime)
    # Calculate y_hat_prime = model(y_hat)
    # minimize ||y_prime - y_hat_prime||^2
    
    y_hat, y_hat_prime = model(y_prime)
    
    # minimize ||y_prime - y_hat_prime||^2
    #print(f"Shapes: y_hat: {y_hat.shape}, y_hat_prime: {y_hat_prime.shape}")
    #print(f"Shapes: y_prime: {y_prime.shape}")
    
    # Calculate the loss
    loss = criterion(y_hat_prime, y_prime)#.unsqueeze(0))
    #print(f"Loss: {loss.item()}")
    
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
    losses.append(loss.item())
    
    #print(f"Reconstruction error for True image: {reconstruction_errors[-1]}")
    #print(f"Error between y_prime and y_hat_prime: {loss}")
    
    if iteration_number == 0:
        # Show the first prediction
        y_hat_ax[2].matshow(y_hat)
        y_hat_fig.canvas.draw()
        y_hat_fig.canvas.flush_events()
        input("Press enter to continue")
    iteration_number += 1
    if iteration_number % 100 == 0:
        print(f"Reconstruction error for True image: {reconstruction_errors[-1]}")
        print(f"Error between y_prime and y_hat_prime: {loss}")
        # Update the figure
        y_hat_ax[2].matshow(y_hat)
        y_hat_fig.canvas.draw()
        y_hat_fig.canvas.flush_events()
        
        # Plot the loss and the reconstruction error
        loss_ax[0].plot(reconstruction_errors)
        loss_ax[1].plot(losses)
        loss_fig.canvas.draw()
        loss_fig.canvas.flush_events()
        plt.pause(0.1)
        if reconstruction_errors[-1] < 0.001:
            break