
from collections import OrderedDict
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
from utils import reconstruct_error
from pytorch_msssim import ssim, ms_ssim
import phantominator as ph
from tqdm import tqdm
from regularization import vector_similarity_regularization, number_of_edges_regularization, binary_regularization

from pytorch_models import SequenceToImageCNN, UNet
import torchviz

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
        self.a = pt.tensor(0.1, device='cuda', requires_grad=True)
        self.parameters = [self.a]
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
        y_hat_prime = self.radon_t.backprojection(s)
        # Scale
        #y_hat_prime = (y_hat_prime - pt.min(y_hat_prime)) / (pt.max(y_hat_prime) - pt.min(y_hat_prime))
        y_hat_prime = y_hat_prime.reshape((1,1,self.output_shape[0],self.output_shape[1]))
        y_hat = self.nn(y_hat_prime)
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

def get_htc_scan(angles = None):
    base_path = "/home/ilmari/python/limited-angle-tomography/HTC_files/"
    htc_file = "tb"
    htc_file = "htc2022_" + htc_file + "_full_recon_fbp_seg.png"
    
    if angles is None:
        angles = np.linspace(0,180,180, endpoint=False)

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
    rt = FBPRadon(img_side_len, angles, clip_to_circle=True)
    sinogram = rt.forward(img)
    #y = rt.backprojection(sinogram)
    return y, sinogram, angles, outer_mask

if __name__ == "__main__":
    
    if True:
        circle = create_circle()
        #phantom = ph.ct_shepp_logan(128)
        y = circle.matrix
        #circle = Circle.from_matrix(phantom)
        #y = phantom
        # Pad and round to 0 or 1
        y = pt.tensor(y, dtype=pt.float32, device='cuda')
        y = pt.nn.functional.pad(y, (0,1,0,1))
        #y = pt.round(y)
        y_np = y.cpu().detach().numpy()
        
        angles = np.linspace(0,60,60, endpoint=False)
        
        # This is where we measure how far from the detector the item (circle) is. IRL, for example laser distance measurement
        cpu_sinogram, dist_front, dist_back = circle.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
        
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
        outer_mask = pt.tensor(outer_mask, dtype=pt.float32, device='cuda', requires_grad=False)
        # Pad to 128x128
        outer_mask = pt.nn.functional.pad(outer_mask, (0,1,0,1))
        # Plot the cpu sinogram
        cpu_sinogram = filter_sinogram_cpu(cpu_sinogram)
        # fliplr
        cpu_sinogram = np.fliplr(cpu_sinogram)
        #plt.matshow(cpu_sinogram)
        #plt.title("CPU sinogram")
        
    
    angles = np.linspace(0,30,60, endpoint=True)
    
    y, s, angles, outer_mask = get_htc_scan(angles=angles)
    #angles = angles.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
    print(f"y shape: {y.shape}")
    #print(f"s shape: {s.shape}")
    print(f"angles shape: {angles.shape}")
    
    
    
    # Create the model
    model = CustomModel(y.shape, angles, outer_mask = outer_mask)
    model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Now s is the sinogram of y, and s_hat is the sinogram of y_hat
    s = model.radon_t.forward(y)
    # Scale the true sinogram
    #s = s / pt.max(s)
    #s = (s - pt.min(s)) / (pt.max(s) - pt.min(s))
    #s = (s - pt.mean(s)) / pt.std(s)

    # Plot the true y, naive reconstruction y_prime, and the NN predicted y
    image_fig, image_axes = plt.subplots(2,2, figsize=(10,5))
    # Plot the true y, and the predicted y
    image_axes[0][0].matshow(y_np)
    image_axes[0][1].matshow(s.cpu().detach().numpy())
    image_axes[0][0].set_title("True image")
    image_axes[0][1].set_title("True Sinogram")
    image_axes[1][0].set_title("Predicted image")
    image_axes[1][1].set_title("Sinogram of predicted image")
    
    naive_rec_fig, naive_rec_ax = plt.subplots()
    y_hat_naive = model.radon_t.backprojection(s)
    y_hat_naive_np = y_hat_naive.cpu().detach().numpy()
    naive_rec_ax.matshow(y_hat_naive_np)
    naive_rec_ax.set_title(f"FBP reconstruction of the true image.")
    
    plt.show(block=False)
    iteration_number = 0
    criterion = pt.nn.MSELoss()

    loss_fig, loss_axes = plt.subplots(1,2, figsize=(10,5))
    loss_axes[0].set_title("Reconstruction error")
    loss_axes[0].set_xlabel("Iteration number")
    loss_axes[0].set_ylabel("Reconstruction error")
    loss_axes[1].set_title("Loss")
    loss_axes[1].set_xlabel("Iteration number")
    loss_axes[1].set_ylabel("Loss")

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

    # Convert the base images to rgb
    base_images = base_images.unsqueeze(1)
    base_images = pt.cat([base_images, base_images, base_images], dim=1)
    print(f"Base images shape: {base_images.shape}")
    def regularization(y_hat):
        #return pt.tensor(0.0, device='cuda', requires_grad=True)
        mat = number_of_edges_regularization(y_hat,coeff=1, filter_sz=3)
        # Return the mean of the matrix
        edge_regularization = pt.mean(mat)*5
        #binary_reg = binary_regularization(y_hat, coeff=0.1)
        return edge_regularization# + binary_reg
        #return pt.tensor(0.0, device='cuda')
        # convert y_hat
        y_hat = y_hat.unsqueeze(0)
        y_hat = pt.cat([y_hat, y_hat, y_hat], dim=0)
        y_hat = y_hat.unsqueeze(0)
        #print(f"y_hat shape: {y_hat.shape}")
        sims = vector_similarity_regularization(y_hat, base_images, coeff=1)
        return pt.mean(sims)

    while True:
        
        y_hat, s_hat = model(s)
        y_hat = y_hat.reshape(y.shape)
        s_hat = s_hat.reshape(s.shape)
        # Scale s_hat
        #s_hat = (s_hat - pt.min(s_hat)) / (pt.max(s_hat) - pt.min(s_hat))
        #s_hat = (s_hat - pt.mean(s_hat)) / pt.std(s_hat)
        # minimize ||s - s_hat||^2
        
        #print(f"Shapes: s_hat: {s_hat.shape}")
        # Calculate the loss
        #mse_loss = pt.tensor(0,dtype=pt.float32,requires_grad = True)
        mse_loss = criterion(s,s_hat)
        #y_hat = y_hat.reshape((1,128,128))
        
        regularization_loss = regularization(y_hat)
        
        loss = mse_loss + regularization_loss
        
        regularization_loss.backward(retain_graph=True)
        
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
        s_hat_np = s_hat.cpu().detach().numpy()

        total_absolute_reconstruction_error = np.mean(np.abs(y_np - y_hat_np))
        # Round y_hat, and calculate a confusion matrix by comparing
        # y_hat and y_np
        y_hat_rounded_np = np.round(y_hat_np)
        M = np.zeros((2,2))
        # M = [[TP, FN], [FP, TN]]
        M[0,0] = np.sum(np.logical_and(y_np == 1, y_hat_rounded_np == 1))
        M[0,1] = np.sum(np.logical_and(y_np == 1, y_hat_rounded_np == 0))
        M[1,0] = np.sum(np.logical_and(y_np == 0, y_hat_rounded_np == 1))
        M[1,1] = np.sum(np.logical_and(y_np == 0, y_hat_rounded_np == 0))
        
        s_score = (M[0,0] * M[1,1] - M[1,0]*M[0,1]) / np.sqrt((M[0,0] + M[0,1]) * (M[1,0] + M[1,1]) * (M[0,0] + M[1,0]) * (M[0,1] + M[1,1]))
        reconstruction_errors.append(3 - 3*s_score)
        
        #print(f"Reconstruction error for True image: {reconstruction_errors[-1]}")
        #print(f"Error between y_prime and y_hat_prime: {loss}")
        
        if iteration_number == 0:
            # Generate compute graph
            torchviz.make_dot(regularization_loss, params=dict(model.named_parameters())).render("compute_graph", format="png")
            # Show the first prediction
            image_axes[1][0].matshow(y_hat_np)
            image_axes[1][1].matshow(s_hat_np)
            image_fig.canvas.draw()
            image_fig.canvas.flush_events()
            input("Press enter to continue")
        iteration_number += 1
        if iteration_number % 100 == 0:
            print(f"Reconstruction error between y and y_hat: {reconstruction_errors[-1]}")
            print(f"Loss between s and s_hat: {mse_loss}")
            # Update the figure
            image_axes[1][0].matshow(y_hat_np)
            image_axes[1][1].matshow(s_hat_np)
            image_fig.canvas.draw()
            image_fig.canvas.flush_events()
            
            # Plot the loss and the reconstruction error
            loss_axes[0].plot(reconstruction_errors)
            loss_axes[1].plot(losses, color='black')
            loss_axes[1].plot(mse_losses, color='red')
            loss_axes[1].plot(regularization_losses, color='blue')
            loss_axes[1].legend(["Total Loss", "Loss function", "Regularization"])
            loss_fig.canvas.draw()
            loss_fig.canvas.flush_events()
            plt.pause(0.1)
            if reconstruction_errors[-1] < 0.001:
                break