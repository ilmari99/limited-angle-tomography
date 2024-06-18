
from collections import OrderedDict
import os
import time

import cv2
import kornia
from AbsorptionMatrices import Circle

import torch as pt

import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from reconstruct import reconstruct_outer_shape
from utils import filter_sinogram
from utils import FBPRadon
from pytorch_msssim import ssim, ms_ssim
import phantominator as ph
from regularization import total_variation_regularization, vector_similarity_regularization, number_of_edges_regularization, binary_regularization
from pytorch_models import HTCModel, SequenceToImageCNN, UNet, UNet2, SinogramCompletionTransformer, LSTMSinogram
import imageio


class ModelBase(nn.Module):
    """
    Takes in a sinogram, and outputs y_hat and s_hat.
    """
    def __init__(self, proj_dim : int, angles, a = 0.1, image_mask = None, device="cuda"):
        super(ModelBase, self).__init__()
        self.dim = proj_dim
        self.angles = np.array(angles)
        self.output_image_shape = (self.dim, self.dim)
        self.device = device
        if image_mask is None:
            image_mask = pt.ones(self.output_image_shape, device=device, requires_grad=False)
        self.image_mask = image_mask
        self.radon_t = FBPRadon(proj_dim, self.angles, a = a, clip_to_circle=False, device=device)
        self._set_step_size_angle()
        self.nn = self.create_nn()
        self.nn.to(self.device)
        self.nn.train()
    
    def _set_step_size_angle(self):

        half_step_size_in_rads = np.deg2rad(0.5)
        full_step_size_in_rads = np.deg2rad(1)
        
        step_len = np.mean(np.diff(self.angles))
        
        assert np.isclose(step_len, half_step_size_in_rads) or np.isclose(step_len, full_step_size_in_rads), f"Step length is not correct. Step length: {step_len}"
        
        # Find whether the angles are in half steps or full steps
        self.step_size_angles = 0.5 if np.isclose(step_len, half_step_size_in_rads) else 1
        return
        
    def parameters(self):
        return self.nn.parameters()
        
    def create_nn(self) -> nn.Module:
        raise NotImplementedError(f"create_nn must be implemented.")
    
class NoModel(nn.Module):
    """ No model, where we just optimize a circle (y_hat) to produce the sinogram.
    """
    def __init__(self, proj_dim, angles, a = 0.1, image_mask=None, device="cuda"):
        self.dim = proj_dim
        self.angles = np.array(angles)
        self.output_image_shape = (self.dim, self.dim)
        self.device = device
        if image_mask is None:
            image_mask = pt.ones(self.output_image_shape, device=device, requires_grad=False)
        # Copy image_mask with gradient
        self.image_mask = image_mask
        self.y_hat = pt.tensor(image_mask, device=device, requires_grad=True)
        
        self.radon_t = FBPRadon(proj_dim, self.angles, a = a, clip_to_circle=False, device=device)
        self._set_step_size_angle()
        self.use_original = False
        super(NoModel, self).__init__()
        
    def _set_step_size_angle(self):
        half_step_size_in_rads = np.deg2rad(0.5)
        full_step_size_in_rads = np.deg2rad(1)
        
        step_len = np.mean(np.diff(self.angles))
        
        assert np.isclose(step_len, half_step_size_in_rads) or np.isclose(step_len, full_step_size_in_rads), f"Step length is not correct. Step length: {step_len}"
        
        # Find whether the angles are in half steps or full steps
        self.step_size_angles = 0.5 if np.isclose(step_len, half_step_size_in_rads) else 1
        return
    
    def parameters(self):
        return [self.y_hat]
    
    def forward(self, s):
        y_hat = self.y_hat
        y_hat = pt.sigmoid(y_hat)
        #y_hat = 1/(1 + pt.exp(-2*(y_hat)))
        y_hat = kornia.filters.bilateral_blur(y_hat.unsqueeze(0).unsqueeze(0),
                                              kernel_size=(3,3),
                                                sigma_color=10.0,
                                                sigma_space=(1.0,1.0))
        y_hat = self.image_mask * y_hat
        y_hat = y_hat.squeeze()
        s_hat = self.radon_t.forward(y_hat)
        return y_hat, s_hat
    
class ReconstructFromSinogram(ModelBase):
    """ This class is for models, that predict the image directly from the sinogram.
    """
    def __init__(self, proj_dim: int, angles, a = 0.1, image_mask=None, device="cuda"):
        self.use_original = True if proj_dim == 560 else False
        super().__init__(proj_dim, angles, a = a, image_mask=image_mask, device=device)
    
    def create_nn(self) -> nn.Module:
        if self.use_original:
            #print("Using original HTC model")
            htc_net = HTCModel((181, self.dim), init_features=128, overwrite_cache=False, init_channels=2, load_weights="htc_trained_model.pth")
        else:
            #print("Using new HTC model")
            htc_net = HTCModel((len(self.angles), self.dim), init_features=32, overwrite_cache=False, init_channels=1)
            
        htc_net.to("cuda")
        #summary(htc_net, (2,181,560), batch_size=1)
        return htc_net
    
    def forward(self, s):
        s = s.float()
        s.to(self.device)
        if self.use_original:
            s = s / 255
            # Pad the sinogram to 181 x 512
            missing_rows = 181 - s.shape[0]
            s = pt.nn.functional.pad(s, (0,0,0,missing_rows))
            s = s.reshape((1,1,181,self.dim))
            #print(f"Sinogram shape: {s.shape}")
            #s = s.reshape((1,1,len(self.angles),self.dim))
            # Create a mask that is 1 where we have data, and 0 where we padded
            mask = pt.ones((181,560), device='cuda', requires_grad=False)
            mask[181-missing_rows:,:] = 0
            #print(f"Sinogram shape: {s.shape}")
            mask = mask.reshape((1,1,181,560))
            #print(f"Mask shape: {mask.shape}")
            
            # Concat mask on channel dimension
            s = pt.cat([s, mask], dim=1)
        else:
            s = s.reshape((1,1,len(self.angles),self.dim))

        #s = s.reshape((1,1,len(self.angles),self.dim))
        y_hat = self.nn(s)
        
        if self.use_original:
            # rotate to angles[0]
            print(f"Rotating yhat ({y_hat.shape}) to {np.rad2deg(self.angles[0])} degrees")
            y_hat = torchvision.transforms.functional.rotate(y_hat, np.rad2deg(self.angles[0]))
            print(f"Rotated yhat shape {y_hat.shape}")
            y_hat = y_hat.reshape((512,512))
        #print(f"Y_hat shape: {y_hat.shape}")
        y_hat = pt.squeeze(y_hat)

        # Pad equally from all sides to dimxdim
        num_missing_cols = self.dim - y_hat.shape[1]
        num_missing_rows = self.dim - y_hat.shape[0]
        row_pad_up = num_missing_rows // 2
        row_pad_down = num_missing_rows - row_pad_up
        col_pad_left = num_missing_cols // 2
        col_pad_right = num_missing_cols - col_pad_left
        y_hat = pt.nn.functional.pad(y_hat, (col_pad_left, col_pad_right, row_pad_up, row_pad_down))
        
        y_hat = y_hat * self.image_mask
        
        #print(f"Yhat passed to radon_t: {y_hat.shape}")
        
        s_hat = self.radon_t.forward(y_hat)
        
        return y_hat, s_hat
    
class PredictSinogramAndReconstruct(ModelBase):
    def __init__(self, proj_dim: int, angles, a = 0.1, image_mask=None, device="cuda"):
        super().__init__(proj_dim, angles, image_mask, device)
        self.radon_t_backward = FBPRadon(self.dim, self.output_sinogram_angles, a = a, clip_to_circle=False, device=device)
    
    def scale(self, y_hat):
        y_hat = y_hat - pt.min(y_hat)
        y_hat = y_hat / pt.max(y_hat)
        return y_hat
    
    def create_nn(self) -> nn.Module:
        num_angles = 180 if self.step_size_angles == 1 else 360
        self.output_sinogram_angles = np.deg2rad(np.linspace(0, 180, num_angles, endpoint=False))
        transformer = LSTMSinogram(
            dim=self.dim,
            input_sinogram_angles = self.angles,
            output_sinogram_angles=self.output_sinogram_angles,
            device=self.device)
        self.nn = transformer
        return transformer
    
    def forward(self, s):
        s = s.float()
        s.to(self.device)
        print(f"Sinogram shape: {s.shape}")
        # Predict the sinogram
        s_hat = self.nn(s)
        
        # Backproject
        y_hat = self.radon_t_backward.backprojection(s_hat)
        y_hat = self.scale(y_hat)
        y_hat = y_hat * self.image_mask
        # Scale
        y_hat = self.scale(y_hat)
        
        # s_hat is the forward of the reconstruction
        s_hat_2 = self.radon_t.forward(y_hat)
        return y_hat, s_hat_2            
    
class BackprojectAndUNet(ModelBase):
    
    def __init__(self, proj_dim: int, angles, a = 0.1, image_mask=None, device="cuda"):
        super().__init__(proj_dim, angles, a = a, image_mask=image_mask, device=device)
    
    def create_nn(self) -> nn.Module:
        nn_ = UNet(in_channels=1, out_channels=1, init_features=32, load_weights="model_gpu.pt")
        self.nn = nn_
        return nn_
        
    def forward(self, s):
        s = s.float()
        s.to(self.device)
        y_hat_prime = self.radon_t.backprojection(s)
        
        # Scale
        #y_hat_prime = (y_hat_prime - pt.min(y_hat_prime)) / (pt.max(y_hat_prime) - pt.min(y_hat_prime))
        y_hat_prime = y_hat_prime.reshape((1,1,self.dim,self.dim))
        y_hat = self.nn(y_hat_prime)

        # Multiply elementwise with outer mask
        y_hat = y_hat * self.image_mask
        
        # s_hat is the sinogram of y_hat, which should be equal to s
        s_hat = self.radon_t.forward(y_hat)
        
        return y_hat, s_hat
    
    
def create_circle():
    circle = Circle(63)
    nholes = 10
    hole_volatility = 0.4
    n_missing_pixels = 0.05
    hole_ratio_limit = 10
    # Create holes in different angles
    for i in range(nholes):
        at_angle = np.random.randint(0,90)
        circle.make_holes(1,
                    n_missing_pixels,
                    hole_volatility,
                    hole_ratio_limit,
                    at_angle)
    return circle

def get_htc_scan(angles, level = 1, sample = "a", return_raw_sinogram = False):
    base_path = "/home/ilmari/python/limited-angle-tomography/htc2022_test_data/"
    # htc2022_01a_recon_fbp_seg.png
    htc_file = f"htc2022_0{level}{sample}_recon_fbp_seg.png"
    sinogram_file = f"htc2022_0{level}{sample}_limited_sinogram.csv"
    angle_file = f"htc2022_0{level}{sample}_angles.csv"
    angle_file = os.path.join(base_path, angle_file)
    
    print(f"Loading sample {level}{sample}",end="")

    # Load angles
    angles = np.loadtxt(angle_file,dtype=np.str_, delimiter=",")
    angles = np.array([float(angle) for angle in angles])
    print(f" at angle: {angles[0]}")

    # Load the img
    img = read_image(os.path.join(base_path, htc_file))
    img = pt.tensor(img, dtype=pt.float32, device='cuda', requires_grad=False)
    #img = img.squeeze()
    # Rotate to angles[0]
    #img = torchvision.transforms.functional.rotate(img, -angles[0])
    img = img.squeeze()
    
    circle = Circle.from_matrix(img.cpu().detach().numpy())
    _, dist_front, dist_back = circle.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
    outer_mask = reconstruct_outer_shape(angles, dist_front, dist_back, zero_threshold=0.1)
    outer_mask = pt.tensor(outer_mask, dtype=pt.float32, device='cuda', requires_grad=False)
    
    print(f"Image shape: {img.shape}")
    #scale
    y = img / 255
    
    if return_raw_sinogram:
        sinogram = np.loadtxt(os.path.join(base_path, sinogram_file), delimiter=',')
        sinogram = pt.tensor(sinogram, dtype=pt.float32, device='cuda') * 255
        return y, outer_mask, sinogram, angles
    
    return y, outer_mask

def get_shepp_logan_scan(angles):
    phantom = ph.ct_shepp_logan(512)
    y = pt.tensor(phantom, dtype=pt.float32, device='cuda')
    #sino = rt.forward(y)
    shape = Circle.from_matrix(phantom)
    _, dist_front, dist_back = shape.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
    outer_mask = reconstruct_outer_shape(angles, dist_front, dist_back, zero_threshold=0.1)
    outer_mask = pt.tensor(outer_mask, dtype=pt.float32, device='cuda', requires_grad=False)
    return y, outer_mask

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
    y = y * outer_mask
    # Make y binary
    y = pt.clip(y, 0, 1)
    y = pt.round(y)
    return y, outer_mask

def show_sinogram(y):
    radon_t_full = FBPRadon(128, np.deg2rad(np.arange(0,180,1)),device="cuda")
    
    s_full = radon_t_full.forward(y)
    
    s_full_np = s_full.cpu().detach().numpy()
    
    fig, ax = plt.subplots(1,2)
    
    ax[0].imshow(s_full_np[0:90,:])
    ax[0].set_title("Sinogram from 0-90 degrees")
    ax[1].imshow(np.flipud(s_full_np[90:,:]))
    ax[1].set_title(f"Sinogram from 90-180 degrees, UD flipped")
    plt.show()
    exit()

def load_base_images(path):
    base_image_paths = list(filter(lambda x: "shape" in x, os.listdir(path)))
    base_image_paths = base_image_paths[0:10]
    # Load the numpy arrays
    base_images = []
    for image_path in base_image_paths:
        base_images.append(np.load(os.path.join(path, image_path)))
        
    # Convert to torch tensors
    base_images = pt.tensor(base_images, dtype=pt.float32, device='cuda')
    print(f"Base images shape: {base_images.shape}")

    # Convert the base images to rgb
    base_images = base_images.unsqueeze(1)
    base_images = pt.cat([base_images, base_images, base_images], dim=1)
    print(f"Base images shape: {base_images.shape}")
    return base_images

def find_good_a(s, sinogram, device='cuda'):
    diffs = []
    sinograms = []
    smallest_error = float("inf")
    smallest_error_a = 0
    for a in np.linspace(0,10,200):
        sinogram_ = filter_sinogram(sinogram, a=a, device=device)
        mae = pt.mean(pt.abs(s - sinogram_))
        diffs.append(mae)
        sinograms.append(sinogram_)
        print(f"a = {a}: {mae}")
        if mae < smallest_error:
            smallest_error = mae
            smallest_error_a = a
    return smallest_error_a

class LPLoss(nn.Module):
    def __init__(self, p=0.5):
        """ Loss that computes the Lp norm of the difference between two images.
        """
        super(LPLoss, self).__init__()
        self.p = p
        
    def forward(self, X, Y):
        # calculate the mean of the Lp norm of the difference between X and Y
        return pt.mean(pt.abs(X - Y)**self.p * (1/self.p))

HTC_LEVEL_TO_ANGLES = {
    7 : np.linspace(0, 30, 60, endpoint=False),
    6 : np.linspace(0, 40, 80, endpoint=False),
    5 : np.linspace(0, 50, 100, endpoint=False),
    4 : np.linspace(0, 60, 120, endpoint=False),
    3 : np.linspace(0, 70, 140, endpoint=False),
    2 : np.linspace(0, 80, 160, endpoint=False),
    1 : np.linspace(0, 90, 180, endpoint=False)
}

if __name__ == "__main__":
    filter_raw_sinogram_with_a = 5.5
    filter_sinogram_of_predicted_image_with_a = 5.5
    S_total_scores = []
    time_limit_s = 30
    image_folder = "BenchmarkReconstructionTVRegNoModelFilt55T30"
    skip_done_tests = False
    trim_sinogram = True
    pad_y_and_mask = False
    compare_s_score_to_unpadded = True
    criterion = LPLoss(p=2.0)
    
    def regularization(y_hat):
        #return pt.tensor(0)
        tv = total_variation_regularization(y_hat,normalize=True)
        return tv
    
    os.makedirs(image_folder, exist_ok=True)
    # Copy this file to the image folder
    os.system(f"cp {__file__} {image_folder}")
    
    for htc_level in range(1,8):
        S_total_scores.append(0)
        for sample in ["a", "b", "c"]:
            
            best_image_name = f"best_image_htc_level_{htc_level}_sample_{sample}.png"
            best_image_path = os.path.join(image_folder, best_image_name)
            
            gif_name = f"reconstruction_images_htc_level_{htc_level}_sample_{sample}.gif"
            gif_path = os.path.join(image_folder, gif_name)
            
            plot_name = f"losses_htc_level_{htc_level}_sample_{sample}.png"
            plot_path = os.path.join(image_folder, plot_name)
            
            # If all the files exist, skip this iteration
            if skip_done_tests and os.path.exists(best_image_path) and os.path.exists(gif_path) and os.path.exists(plot_path):
                continue

            angles = HTC_LEVEL_TO_ANGLES[htc_level]
            y, image_mask, sinogram, angles = get_htc_scan(angles=angles, level=htc_level, sample=sample, return_raw_sinogram=True)
            
            assert not (trim_sinogram and pad_y_and_mask), "Cannot trim sinogram and pad y and mask"
            if trim_sinogram:
                # Trim the sinogram to have the same number of columns as y,
                # because why would the sinogram have more columns than the image?
                num_extra_cols = sinogram.shape[1] - y.shape[1]
                if num_extra_cols != 0:
                    rm_from_left = num_extra_cols // 2 - 1
                    rm_from_right = num_extra_cols - rm_from_left
                    sinogram = sinogram[:, rm_from_left:sinogram.shape[1]-rm_from_right]
                    print(f"Trimmed sinogram to shape: {sinogram.shape}")
                    
            if pad_y_and_mask:
                # Pad y to the number of columns in the sinogram
                num_missing_rows = sinogram.shape[1] - y.shape[0]
                num_missing_cols = sinogram.shape[1] - y.shape[1]
                row_pad_up = num_missing_rows // 2
                row_pad_down = num_missing_rows - row_pad_up
                col_pad_left = num_missing_cols // 2
                col_pad_right = num_missing_cols - col_pad_left
                y_not_padded = y
                y = pt.nn.functional.pad(y, (col_pad_left, col_pad_right, row_pad_up, row_pad_down))
                image_mask = pt.nn.functional.pad(image_mask, (col_pad_left, col_pad_right, row_pad_up, row_pad_down))
                print(f"Padded y and mask from {y_not_padded.shape} to {y.shape}")

            #model = ReconstructFromSinogram(sinogram.shape[1],
            #                    np.deg2rad(angles),
            #                    a=filter_sinogram_of_predicted_image_with_a,
            #                    image_mask=image_mask,
            #                    device='cuda'
            #                    )
            model = NoModel(sinogram.shape[1], np.deg2rad(angles), a=filter_sinogram_of_predicted_image_with_a, image_mask=image_mask, device='cuda')
            
            #model = PredictSinogramAndReconstruct(128, np.deg2rad(angles), image_mask=outer_mask, device='cuda')
            #model = BackprojectAndUNet(sinogram.shape[1], np.deg2rad(angles), a=filter_sinogram_of_predicted_image_with_a, image_mask=image_mask, device='cuda')
            model.to('cuda')
            optimizer = optim.Adam(model.parameters(), lr=0.9, amsgrad=True)
            
            raw_sinogram_mean = pt.mean(sinogram)
            raw_sinogram_std = pt.std(sinogram)
            raw_sinogram = sinogram
            
            if filter_raw_sinogram_with_a != 0:
                filtered_sinogram = filter_sinogram(raw_sinogram,filter_raw_sinogram_with_a,device="cuda")
            else:
                filtered_sinogram = raw_sinogram
        
            filtered_sinogram_mean = pt.mean(filtered_sinogram)
            filtered_sinogram_std = pt.std(filtered_sinogram)

            y_np = y.cpu().detach().numpy()

            iteration_number = 0

            s_errors = []
            mse_losses = []
            regularization_losses = []
            losses = []
            
            t_start = time.time()
            
            images = []
            image_of_best_reconstruction = None
            best_s_score = -1
            
            input_sinogram = raw_sinogram
            while (time.time() - t_start < time_limit_s) and (not model.use_original or (iteration_number < 1)):
                
                print(f"Iteration {iteration_number}", end="\r")
                y_hat, s_hat = model(input_sinogram)
                y_hat = y_hat.reshape(y.shape)
                s_hat = s_hat.reshape(input_sinogram.shape)
                
                # Calculate the loss between the
                # sinogram of the predicted object, and the filtered sinogram
                mse_loss = criterion(filtered_sinogram,s_hat)
                
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
                s_hat_np = s_hat.cpu().detach().numpy()

                M = np.zeros((2,2))
                if compare_s_score_to_unpadded and pad_y_and_mask:
                    print(f"Using original Y, and not the padded version")
                    y_not_padded_np = y_not_padded.cpu().detach().numpy()
                    y_hat_np = y_hat_np[row_pad_up:-row_pad_down, col_pad_left:-col_pad_right]
                    y_np = y_not_padded_np
                    
                y_hat_np = np.clip(y_hat_np, 0, 1)
                y_hat_np_rounded = np.round(y_hat_np)

                # M = [[TP, FN], [FP, TN]]
                M[0,0] = np.sum(np.logical_and(y_np == 1, y_hat_np_rounded == 1))
                M[0,1] = np.sum(np.logical_and(y_np == 1, y_hat_np_rounded == 0))
                M[1,0] = np.sum(np.logical_and(y_np == 0, y_hat_np_rounded == 1))
                M[1,1] = np.sum(np.logical_and(y_np == 0, y_hat_np_rounded == 0))
                
                # MCC
                s_score = (M[0,0] * M[1,1] - M[1,0]*M[0,1]) / np.sqrt((M[0,0] + M[0,1]) * (M[1,0] + M[1,1]) * (M[0,0] + M[1,0]) * (M[0,1] + M[1,1]))
                s_errors.append(s_score)
                
                iteration_number += 1
                
                if iteration_number % 30 == 0: 
                    images.append(y_hat_np)
                if s_score > best_s_score:
                    best_s_score = s_score
                    image_of_best_reconstruction = y_hat_np_rounded
                
            # After 1 minute, we find the lowest loss
            min_loss = min(losses)
            min_loss_idx = losses.index(min_loss)
            selected_s_score = s_errors[min_loss_idx]
            print(f"HTC level: {htc_level}, sample: {sample}, s_score: {selected_s_score}")
            S_total_scores[-1] += selected_s_score
            
            # Save the best image
            plt.imsave(best_image_path, image_of_best_reconstruction, cmap='gray')
            
            if len(images) > 1:
                # Save the images as a gif
                images = [255*img for img in images]
                imageio.mimsave(gif_path, images)
                
                # Plot the losses
                fig, ax = plt.subplots(2,2)
                fig.suptitle(f"HTC level {htc_level}, sample {sample}")
                ax[0][0].plot(mse_losses)
                ax[0][0].set_title("Loss between sinograms")
                ax[0][1].plot(regularization_losses)
                ax[0][1].set_title("Regularization loss")
                ax[1][0].plot(losses)
                ax[1][0].set_title("Total loss")
                ax[1][1].plot(s_errors)
                ax[1][1].set_title("S score")
                
                # Save the plot
                plt.savefig(plot_path)
            plt.close()
        print(f"HTC level {htc_level} total s_score: {S_total_scores[-1]}")
        
    print(f"All scores: {S_total_scores}")
            
            
            
            