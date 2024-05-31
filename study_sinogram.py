
import os
from AbsorptionMatrices import Circle

import torch as pt
from torch_radon import Radon

import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt
from reconstruct import filter_sinogram as filter_sinogram_cpu
from reconstruct import reconstruct_outer_shape
import phantominator as ph

from pytorch_models import SequenceToImageCNN, UNet

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
    filtSino = pt.zeros((projLen, numAngles), device='cuda', requires_grad=False)
    
    for i in range(numAngles):
        projfft = pt.fft.fft(sino[:, i])
        filtProj = projfft * filt
        ifft_filtProj = pt.fft.ifft(filtProj)
        
        filtSino[:, i] = pt.real(ifft_filtProj)

    return filtSino.T
        
class FBPRadon(Radon):
    def __init__(self, *args, **kwargs):
        super(FBPRadon, self).__init__(*args, **kwargs)
        self.a = 0.1
        
    def forward(self, x):
        s = super().forward(x)
        s = filter_sinogram(s, a = self.a)
        return s
    

    
class CustomModel(nn.Module):
    def __init__(self, output_shape, angles, outer_mask = None):
        super(CustomModel, self).__init__()
        if outer_mask is None:
            outer_mask = pt.ones(output_shape, device='cuda', requires_grad=False)
        self.outer_mask = outer_mask
        self.angles = angles
        self.output_shape = output_shape
        self.angles_rad = np.deg2rad(angles)
        self.radon_t = FBPRadon(output_shape[0], self.angles_rad, clip_to_circle=False)
        # Load keras model from 'blurry_to_real_model2.keras'
        #self.nn = SequenceToImageCNN(input_shape = (1, len(angles), output_shape[0]), output_shape = output_shape, hidden_size=512, num_layers=1)
        self.nn = UNet(in_channels=1, out_channels=1, init_features=32)
        print(f"Neural net: {self.nn}")
        # Load weights from keras model
        #self.nn.load_state_dict(pt.load("model_gpu.pt"))
        self.nn.to('cuda')
        self.nn.train()
    
    def parameters(self):
        return self.nn.parameters()
        
    def forward(self, s):
        s = s.float().cuda()
        s = s.reshape((1, len(self.angles), self.output_shape[0]))
        #y_hat_prime = self.radon_t.backprojection(s)
        # Scale
        #y_hat_prime = (y_hat_prime - pt.min(y_hat_prime)) / (pt.max(y_hat_prime) - pt.min(y_hat_prime))
        #y_hat_prime = y_hat_prime.reshape((1,1,self.output_shape[0],self.output_shape[1]))
        y_hat = self.nn(s)
        #print(f"y_hat shape: {y_hat.shape}")
        y_hat = y_hat.reshape(self.output_shape)
        # Multiply elementwise with outer mask
        y_hat = y_hat * self.outer_mask
        # Scale to [0,1]
        #y_hat = (y_hat - pt.min(y_hat)) / (pt.max(y_hat) - pt.min(y_hat))
        
        # s_hat is the sinogram of y_hat, which should be equal to s
        s_hat = self.radon_t.forward(y_hat)
        s_hat = s_hat.reshape(s.shape)
        #s_hat = s_hat / pt.max(s_hat)
        # Scale to [0,1]
        #s_hat = (s_hat - pt.min(s_hat)) / (pt.max(s_hat) - pt.min(s_hat))
        return y_hat, s_hat
    
    
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

def get_htc_scan(angles):
    base_path = "/home/ilmari/python/limited-angle-tomography/HTC_files/"
    htc_file = "tb"
    htc_file = "htc2022_" + htc_file + "_full_recon_fbp_seg.png"

    # Load the img
    img = read_image(os.path.join(base_path, htc_file))
    img = pt.tensor(img, dtype=pt.float32, device='cuda', requires_grad=False)
    img = img.squeeze()
    circle = Circle.from_matrix(img.cpu().detach().numpy())
    _, dist_front, dist_back = circle.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
    outer_mask = reconstruct_outer_shape(angles, dist_front, dist_back, zero_threshold=0.1)
    outer_mask = pt.tensor(outer_mask, dtype=pt.float32, device='cuda', requires_grad=False)
    
    print(f"Image shape: {img.shape}")
    #scale
    y = img / 255
    img_side_len = img.shape[0]
    #rt = FBPRadon(img_side_len, angles, clip_to_circle=True)
    #sinogram = rt.forward(y)
    #y = rt.backprojection(sinogram)
    return y, outer_mask

def get_shepp_logan_scan(angles):
    phantom = ph.ct_shepp_logan(128)
    rt = FBPRadon(128, angles, clip_to_circle=True)
    y = pt.tensor(phantom, dtype=pt.float32, device='cuda')
    #sino = rt.forward(y)
    shape = Circle.from_matrix(phantom)
    _, dist_front, dist_back = shape.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
    outer_mask = reconstruct_outer_shape(angles, dist_front, dist_back, zero_threshold=0.1)
    outer_mask = pt.tensor(outer_mask, dtype=pt.float32, device='cuda', requires_grad=False)
    return y, angles, outer_mask

def get_basic_circle_scan(angles):
    if angles is None:
        angles = np.linspace(0,60,60, endpoint=False)
    circle = create_circle()
    y = circle.matrix
    y = pt.tensor(y, dtype=pt.float32, device='cuda')
    y = pt.nn.functional.pad(y, (0,1,0,1))

    _, dist_front, dist_back = circle.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
    outer_mask = reconstruct_outer_shape(angles, dist_front, dist_back, zero_threshold=0.1)
    outer_mask = pt.tensor(outer_mask, dtype=pt.float32, device='cuda', requires_grad=False)
    outer_mask = pt.nn.functional.pad(outer_mask, (0,1,0,1))
    rt = FBPRadon(y.shape[0], np.deg2rad(angles), clip_to_circle=False)
    sinogram = rt.forward(y)
    return y, outer_mask, sinogram

if __name__ == "__main__":

    angles = np.linspace(0,180,180, endpoint=False)
    
    y, outer_mask, sinogram = get_basic_circle_scan(angles=angles)
    #y, outer_mask = get_htc_scan(angles=angles)
    #y, outer_mask = get_shepp_logan_scan(angles=angles)
    #angles = angles.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
    print(f"y shape: {y.shape}")
    #print(f"s shape: {s.shape}")
    print(f"angles shape: {angles.shape}")

    # Plot the image, sinogram, and fourier transform of the sinogram
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].imshow(y_np)
    ax[0].set_title("Image")
    sinogram_np = sinogram.cpu().detach().numpy()
    ax[1].imshow(sinogram_np)
    ax[1].set_title("Sinogram")
    
    # Compare the similarity of each row of the sinogram to each other
    similarities = np.zeros((sinogram_np.shape[0], sinogram_np.shape[0]))
    for i in range(sinogram_np.shape[0]):
        for j in range(i, sinogram_np.shape[0]):
            # Calculate the cosine similarity between the two rows
            si = sinogram_np[i]
            sj = sinogram_np[j]
            similarities[i,j] = np.dot(si, sj) / (np.linalg.norm(si) * np.linalg.norm(sj))
            
    ax[2].imshow(similarities)
    ax[2].set_title("Similarity of sinogram rows")
    plt.show()