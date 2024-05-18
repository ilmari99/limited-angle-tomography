
from collections import OrderedDict
import os
from AbsorptionMatrices import Circle

import torch as pt
from torch_radon import Radon

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from reconstruct import filter_sinogram, reconstruct_outer_shape
from utils import reconstruct_error
from pytorch_msssim import ssim, ms_ssim
import phantominator as ph

from tqdm import tqdm

def filter_sinogram(sino, a=0.1):
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
    
    projLen, numAngles = sino.shape
    step = 2 * np.pi / projLen
    w = pt.arange(-np.pi, np.pi, step, device='cuda')
    if len(w) < projLen:
        w = pt.cat([w, w[-1] + step])
    
    rn1 = abs(2 / a * pt.sin(a * w / 2))
    rn2 = pt.sin(a * w / 2) / (a * w / 2)
    r = rn1 * (rn2) ** 2
    filt = pt.fft.fftshift(r)
    filt[0] = 0
    filtSino = pt.zeros((projLen, numAngles), device='cuda')
    
    for i in range(numAngles):
        projfft = pt.fft.fft(sino[:, i])
        filtProj = projfft * filt
        ifft_filtProj = pt.fft.ifft(filtProj)
        
        filtSino[:, i] = pt.real(ifft_filtProj)

    return filtSino.T

def decompose_and_reconstruct(matrix, radon_t) -> pt.Tensor:
    """
    Decompose a matrix into a sinogram, and then reconstruct it back.
    """
    sinogram = radon_t.forward(matrix)
    sinogram = filter_sinogram(sinogram)
    
    reconstruction = radon_t.backprojection(sinogram)
    
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
    
class CustomModel(nn.Module):
    def __init__(self, input_shape, angles, outer_mask = None):
        super(CustomModel, self).__init__()
        if outer_mask is None:
            outer_mask = pt.ones(input_shape, device='cuda')
        self.angles = angles
        self.angles_rad = np.deg2rad(angles)
        self.radon_t = Radon(input_shape[0], self.angles_rad, clip_to_circle=False)
        # Load keras model from 'blurry_to_real_model2.keras'
        self.u_net = UNet(in_channels=1, out_channels=1, init_features=32, image_size=input_shape[0])
        # Load weights from keras model
        self.u_net.load_state_dict(pt.load("model_gpu.pt"))
        self.u_net.to('cuda')
        self.u_net.train()
        #print(self.u_net)
            
        self.decompose_and_reconstruct = lambda x: decompose_and_reconstruct(x, self.radon_t)
        
    def forward(self, x):
        x = x.float().cuda()
        y_hat = self.u_net(x)
        # Multiply elementwise with outer mask
        y_hat = y_hat * outer_mask
        # Scale to [0,1]
        y_hat = (y_hat - pt.min(y_hat)) / (pt.max(y_hat) - pt.min(y_hat))
        y_hat_prime = self.decompose_and_reconstruct(y_hat)
        y_hat_prime = y_hat_prime.reshape((128,128))
        # Apply mask and scale to [0,1]
        y_hat_prime = y_hat_prime * outer_mask
        y_hat_prime = (y_hat_prime - pt.min(y_hat_prime)) / (pt.max(y_hat_prime) - pt.min(y_hat_prime))
        return y_hat, y_hat_prime
    
def ssim_regularization(y_hat, base_images, coeff=0.01):
    """
    Calculate the SSIM regularization loss.
    """
    y_hat_replicated = pt.broadcast_to(y_hat, base_images.shape)
    ssim_loss = 1 - ssim(y_hat_replicated, base_images, data_range=1, size_average=False)#, nonnegative_ssim=True, win_size=15, win_sigma=7.5)
    return pt.mean(ssim_loss) * coeff

def binary_regularization(y_hat):
    """ Penalize images based on the number of pixels that are not 0.05 < x < 0.95 """
    y_hat = y_hat.reshape((128,128))
    n_pixels = 128*128
    # Get a mask
    gt_zero = y_hat > 0.05
    lt_one = y_hat < 0.95
    non_binary_pixels = pt.logical_and(gt_zero, lt_one)
    n_non_binary_pixels = pt.sum(non_binary_pixels)
    return n_non_binary_pixels / n_pixels

def histogram_regularization(y_hat, true_y):
    """ Penalize based on how different the histogram of y_hat is from the histogram of true_y
    """
    y_hat = y_hat.reshape((128,128))
    true_y = true_y.reshape((128,128))
    y_hat_hist = pt.histc(y_hat, bins=10, min=0, max=1)
    true_y_hist = pt.histc(true_y, bins=10, min=0, max=1)
    # Compare the distributions using chi-squared distance
    chi_squared = pt.sum((y_hat_hist - true_y_hist)**2 / (y_hat_hist + true_y_hist))
    # Scale to be between 0 and 1 (so 1 == 1000)
    return pt.clip(chi_squared / 1000, 0, 1)
    
    
def create_circle():
    circle = Circle(63)
    nholes = 10
    hole_volatility = 0.5
    n_missing_pixels = 0.4
    hole_ratio_limit = 10
    at_angle = 0#np.random.randint(0,90)
    circle.make_holes(nholes,
                  n_missing_pixels,
                  hole_volatility,
                  hole_ratio_limit,
                  at_angle)
    return circle

if __name__ == "__main__":
    
    #circle = create_circle()
    phantom = ph.ct_shepp_logan(128)
    #y = circle.matrix
    circle = Circle.from_matrix(phantom)
    y = phantom
    # Pad and round to 0 or 1
    y = pt.tensor(y, dtype=pt.float32, device='cuda')
    #y = pt.nn.functional.pad(y, (0,1,0,1))
    #y = pt.round(y)
    y_np = y.cpu().detach().numpy()
    
    angles = np.linspace(0,45,45, endpoint=False)
    
    # This is where we measure how far from the detector the item (circle) is. IRL, for example laser distance measurement
    _, dist_front, dist_back = circle.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
    
    # We can construct a mask, that is 1 where where we know the item is, and 0 where we know it is not.
    # This is the outer shape of the item.
    # 0000000000
    # 0000110000
    # 0011111100
    # 0111111110
    # 0111111110
    # 0011111100
    # 0000110000
    # 0000000000
    outer_mask = reconstruct_outer_shape(angles, dist_front, dist_back, zero_threshold=0.1)
    outer_mask = pt.tensor(outer_mask, dtype=pt.float32, device='cuda')
    # Pad to 128x128
    #outer_mask = pt.nn.functional.pad(outer_mask, (0,1,0,1))
    
    # Create the model
    model = CustomModel((128,128), angles, outer_mask = outer_mask)
    model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #y_ = pt.tensor(y, dtype=pt.float32, device='cuda', requires_grad=True)
    y_prime = decompose_and_reconstruct(y, model.radon_t)
    # Multiply elementwise with outer mask, since we know the outer shape
    y_prime = y_prime * outer_mask
    # Scale to [0,1]
    y_prime = (y_prime - pt.min(y_prime)) / (pt.max(y_prime) - pt.min(y_prime))

    # Plot the true y, naive reconstruction y_prime, and the NN predicted y
    y_hat_fig, y_hat_ax = plt.subplots(2,2, figsize=(10,5))
    # Plot the true y, and the predicted y
    y_hat_ax[0][0].matshow(y_np)
    y_hat_ax[0][1].matshow(y_prime.cpu().detach().numpy())
    y_hat_ax[0][0].set_title("True image")
    y_hat_ax[0][1].set_title("Naive reconstruction from Sinogram")
    y_hat_ax[1][0].set_title("Predicted image")
    y_hat_ax[1][1].set_title("Reconstruction of predicted image from Sinogram")
    plt.show(block=False)
    criterion = nn.MSELoss(reduction="sum")
    iteration_number = 0


    loss_fig, loss_ax = plt.subplots(1,2, figsize=(10,5))
    loss_ax[0].set_title("Reconstruction error")
    loss_ax[0].set_xlabel("Iteration number")
    loss_ax[0].set_ylabel("Reconstruction error")
    loss_ax[1].set_title("Loss")
    loss_ax[1].set_xlabel("Iteration number")
    loss_ax[1].set_ylabel("Loss")


    reconstruction_errors = []
    mse_losses = []
    regularization_losses = []
    losses = []

    base_image_paths = list(filter(lambda x: "shape" in x, os.listdir("Circles128x128_1000")))
    base_image_paths = base_image_paths[0:10]
    # Load the numpy arrays
    base_images = []
    for image_path in base_image_paths:
        base_images.append(np.load(os.path.join("Circles128x128_1000", image_path)))
        
    # Convert to torch tensors
    base_images = pt.tensor(base_images, dtype=pt.float32, device='cuda')
    print(f"Base images shape: {base_images.shape}")
    
    # Calculate the average similarity between the base images to establish a baseline
    similarities = []
    for i in range(len(base_images)):
        for j in range(i+1, len(base_images)):
            bi = base_images[i].unsqueeze(0).unsqueeze(0)
            bj = base_images[j].unsqueeze(0).unsqueeze(0)
            sim = ssim(bi, bj, data_range=1, size_average=False)
            similarities.append(sim)
            print(f"SSIM between base images {i} and {j}: {sim}")
    mean_sim = pt.mean(pt.tensor(similarities))
    print(f"Average SSIM between base images: {mean_sim}")
    

    
    def _regularization(x):
        ssim_reg = ssim_regularization(x.unsqueeze(0), base_images.unsqueeze(0), coeff=1)
        sim = 1 - ssim_reg
        # We want the solution (x) to be similar with the base images
        # so a circle with rectangles missing.
        # If ssim_reg is smaller than 1 - mean_sim, we stop regularizing
        # Else, we regularize with the ssim_reg
        #print(f"SSIM regularization: {ssim_reg}")
        coeff = pt.exp(1*(sim - 1))
        #coeff = 0.001
        return ssim_reg * coeff
    
    def regularization(x):
        return pt.tensor(0, dtype=pt.float32, device='cuda')
        return binary_regularization(x)
    
    def regularization(x):
        return histogram_regularization(x, y)

    while True:
        # Calculate y_hat = decompose_and_reconstruct(y_prime)
        # Calculate y_hat_prime = model(y_hat)
        # minimize ||y_prime - y_hat_prime||^2
        #print(f"Y prime shape: {y_prime.shape}")
        y_hat, y_hat_prime = model(y_prime)
        y_hat = y_hat.reshape((128,128))
        y_hat_prime = y_hat_prime.reshape((128,128))
        # minimize ||y_prime - y_hat_prime||^2
        #print(f"Shapes: y_hat: {y_hat.shape}, y_hat_prime: {y_hat_prime.shape}")
        #print(f"Shapes: y_prime: {y_prime.shape}")
        
        # Calculate the loss
        mse_loss = criterion(y_hat_prime, y_prime)
        #y_hat = y_hat.reshape((1,128,128))
        
        regularization_loss = regularization(y_hat)
        
        loss = mse_loss + regularization_loss
        
        mse_losses.append(mse_loss.item())
        regularization_losses.append(regularization_loss.item())
        losses.append(loss.item())
        
        # Update the model
        # Retain grads
        optimizer.zero_grad()
        loss.backward(retain_graph=True)    
        optimizer.step()        
        
        # Squeeze and detach
        y_hat_np = y_hat.cpu().detach().numpy()
        y_hat_prime_np = y_hat_prime.cpu().detach().numpy()
        total_absolute_reconstruction_error = np.sum(np.abs(y_np - y_hat_np))
        reconstruction_errors.append(total_absolute_reconstruction_error)
        
        #print(f"Reconstruction error for True image: {reconstruction_errors[-1]}")
        #print(f"Error between y_prime and y_hat_prime: {loss}")
        
        if iteration_number == 0:
            # Show the first prediction
            y_hat_ax[1][0].matshow(y_hat_np)
            y_hat_ax[1][1].matshow(y_hat_prime_np)
            y_hat_fig.canvas.draw()
            y_hat_fig.canvas.flush_events()
            input("Press enter to continue")
        iteration_number += 1
        if iteration_number % 100 == 0:
            print(f"Reconstruction error for True image: {reconstruction_errors[-1]}")
            print(f"Error between y_prime and y_hat_prime: {loss}")
            # Update the figure
            y_hat_ax[1][0].matshow(y_hat_np)
            y_hat_ax[1][1].matshow(y_hat_prime_np)
            y_hat_fig.canvas.draw()
            y_hat_fig.canvas.flush_events()
            
            # Plot the loss and the reconstruction error
            loss_ax[0].plot(reconstruction_errors)
            loss_ax[1].plot(losses, color='black')
            loss_ax[1].plot(mse_losses, color='red')
            loss_ax[1].plot(regularization_losses, color='blue')
            loss_ax[1].legend(["Total Loss", "Loss function", "Regularization"])
            loss_fig.canvas.draw()
            loss_fig.canvas.flush_events()
            plt.pause(0.1)
            if reconstruction_errors[-1] < 0.001:
                break