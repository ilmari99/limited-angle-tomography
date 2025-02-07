import itertools
import json
import os
import subprocess
import sys

import imageio
import pandas as pd
from AbsorptionMatrices import Circle
import torch as pt
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image, ImageReadMode
import numpy as np
import matplotlib.pyplot as plt
from reconstruct import reconstruct_outer_shape

from skimage.filters import threshold_multiotsu

from utils import (filter_sinogram,
                   FBPRadon,
                   PatchAutoencoder,
                   extract_patches_2d_pt,
                   reconstruct_from_patches_2d_pt
                   )

from pytorch_msssim import ssim, ms_ssim
import phantominator as ph
from regularization import (create_autoencoder_regularization,
                            create_autoencoder_regularization_diff,
                            binary_regularization,
                            total_variation_regularization,
                            tikhonov_regularization,
                            )

from pytorch_models import (UNet,
                            HTCModel,
                            SinogramToPatchesConnected,
                            EncoderDecoder,
                            )

from torchsummary import summary
import time
import torchvision
import torchviz
from IPython.display import display, clear_output
import __main__ as MAIN_MODULE

class ReconstructorBase(nn.Module):
    """
    Takes in a sinogram, and outputs y_hat and s_hat.
    This level should handle all data formatting, setting the image_mask, and edge_masks.
    """
    def __init__(self, proj_dim : int,
                 angles,
                 a = 0.1,
                 image_mask = None,
                 device=None,
                 edge_pad_size=5,
                 ):
        super(ReconstructorBase, self).__init__()
        self.dim = proj_dim
        self.angles = np.array(angles)
        self.output_image_shape = (self.dim, self.dim)
        self.device = device
        if image_mask is None:
            image_mask = pt.ones(self.output_image_shape, device=device, requires_grad=False)
        self.image_mask = image_mask
        self.edge_pad_mask = self.get_edge_padding_mask(image_mask, pad_size=edge_pad_size)
        self.radon_t = FBPRadon(proj_dim, self.angles, a = a, clip_to_circle=False, device=device)
        #self._set_step_size_angle()
        
    def get_edge_padding_mask(self, image_mask, pad_size=30):
        """ Returns a mask of size dim, dim.
        """
        if image_mask is None or pad_size==0:
            return pt.zeros((self.dim, self.dim), device='cuda', requires_grad=False)
        # Scale the mask down by pad_size
        scaled_down_image_mask = pt.nn.functional.interpolate(image_mask.unsqueeze(0).unsqueeze(0), size=(self.dim-pad_size, self.dim-pad_size), mode='nearest')
        scaled_down_image_mask = scaled_down_image_mask.squeeze()
        print(f"Scaled down image mask shape: {scaled_down_image_mask.shape}")
        # Pad to dim, dim
        pu_pl = pad_size // 2
        pd_pr = pad_size - pu_pl
        scaled_down_image_mask_padded = pt.nn.functional.pad(scaled_down_image_mask, (pu_pl, pd_pr, pu_pl, pd_pr))
        print(f"Scaled down image mask padded shape: {scaled_down_image_mask_padded.shape}")
        # Now, to get the edge_padding_mask, we take a zerto matrix, and set all pixels to 1,
        # where the scaled_down_image_mask_padded is 0 AND where the original image_mask is 1
        edge_pad_mask = pt.zeros((self.dim, self.dim), device='cuda', requires_grad=False)
        edge_pad_mask[(scaled_down_image_mask_padded == 0) & (image_mask == 1)] = 1
        return edge_pad_mask
    
    def _set_step_size_angle(self):

        half_step_size_in_rads = np.deg2rad(0.5)
        full_step_size_in_rads = np.deg2rad(1)
        
        step_len = np.mean(np.diff(self.angles))
        
        assert np.isclose(step_len, half_step_size_in_rads) or np.isclose(step_len, full_step_size_in_rads), f"Step length is not correct. Step length: {step_len}"
        
        # Find whether the angles are in half steps or full steps
        self.step_size_angles = 0.5 if np.isclose(step_len, half_step_size_in_rads) else 1
        return
    
    def forward(self, s):
        raise NotImplementedError("Forward method not implemented")
    
    def parameters(self):
        raise NotImplementedError("Parameters method not implemented")

class NoModel(ReconstructorBase):
    """ No model, where we just optimize a collection of weights in a matrix
    (image of a filled circle) to produce the sinogram.
    """
    def __init__(self, proj_dim, angles, a = 0.1, image_mask=None, device="cuda", edge_pad_size=5, scale_sinogram=False):
        super(NoModel, self).__init__(proj_dim, angles, a = a, image_mask=image_mask, device=device, edge_pad_size=edge_pad_size)
        if scale_sinogram:
            raise NotImplementedError("Scaling sinogram not implemented for NoModel")
        self.weights = pt.tensor(self.image_mask, device=self.device, requires_grad=True)
    
    def parameters(self):
        return [self.weights]
    
    def forward(self, s):
        y_hat = self.weights
        y_hat = pt.sigmoid(y_hat)
        y_hat = self.image_mask * y_hat
        # Set all pixels that are in the edge_pad to 1
        y_hat = pt.where(self.edge_pad_mask == 1, pt.tensor(1.0, device='cuda'), y_hat)
        y_hat = y_hat.squeeze()
        s_hat = self.radon_t.forward(y_hat)
        return y_hat, s_hat

class EncoderDecoderCNNReconstructor(ReconstructorBase):
    def __init__(self, proj_dim, angles, latent_image_side_len=16, encoder_filters = [8,16,32], decoder_filters=[32,16,8], a = 0.1, image_mask=None, device="cuda", edge_pad_size=5, scale_sinogram=False):
        super(EncoderDecoderCNNReconstructor, self).__init__(proj_dim, angles, a = a, image_mask=image_mask, device=device, edge_pad_size=edge_pad_size)
        if scale_sinogram:
            raise NotImplementedError("Scaling sinogram not implemented for EncDec model")
        self.nn = EncoderDecoder((len(angles),proj_dim),
                                 angles=angles,
                                 latent_image_side_len=latent_image_side_len,
                                 output_side_len=proj_dim,
                                 encoder_filters=encoder_filters,
                                 decoder_filters=decoder_filters)
        #print(summary(self.nn,input_size=(1,len(angles),proj_dim)))
        self.latent_image_side_len = latent_image_side_len
        
    def forward(self, sinogram):
        y_hat = self.nn(sinogram)
        y_hat = pt.sigmoid(y_hat)
        y_hat = self.image_mask * y_hat
        # Set all pixels that are in the edge_pad to 1
        y_hat = pt.where(self.edge_pad_mask == 1, pt.tensor(1.0, device='cuda'), y_hat)
        y_hat = y_hat.squeeze()
        s_hat = self.radon_t.forward(y_hat)
        return y_hat, s_hat
    
    def parameters(self):
        return self.nn.parameters()
        
class HTCModelReconstructor(ReconstructorBase):
    """ This class is for models, that predict the image directly from the sinogram.
    """
    def __init__(self, proj_dim: int, angles, a = 0.1, image_mask=None, device="cuda", edge_pad_size=5, init_features=32):
        self.use_original = True if proj_dim == 560 else False
        self.init_features = init_features
        super().__init__(proj_dim, angles, a = a, image_mask=image_mask, device=device, edge_pad_size=edge_pad_size)
        self.nn = self.create_nn()
        
    def create_nn(self) -> nn.Module:
        if self.use_original:
            print("Using original HTC model")
            htc_net = HTCModel((181, self.dim), init_features=self.init_features, overwrite_cache=False, init_channels=2)
        else:
            print("Using new HTC model")
            htc_net = HTCModel((len(self.angles), self.dim), init_features=self.init_features, overwrite_cache=False, init_channels=1)
            
        htc_net.to("cuda")
        return htc_net
    
    def parameters(self):
        return self.nn.parameters()
    
    def forward(self, s):
        s = s.float()
        s.to(self.device)
        if self.use_original:
            s = s / 255
            # Pad the sinogram to 181 x 512
            missing_rows = 181 - s.shape[0]
            s = pt.nn.functional.pad(s, (0,0,0,missing_rows))
            s = s.reshape((1,1,181,self.dim))
            # Create a mask that is 1 where we have data, and 0 where we padded
            mask = pt.ones((181,560), device='cuda', requires_grad=False)
            mask[181-missing_rows:,:] = 0
            mask = mask.reshape((1,1,181,self.dim))
            # Concat mask on channel dimension
            s = pt.cat([s, mask], dim=1)
        else:
            s = s.reshape((1,1,len(self.angles),self.dim))
        y_hat = self.nn(s)
        
        if self.use_original:
            # rotate to angles[0]
            y_hat = torchvision.transforms.functional.rotate(y_hat, np.rad2deg(self.angles[0]))
            y_hat = y_hat.reshape((512,512))
        
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
        # Set all pixels that are in the edge_pad to 1
        y_hat = pt.where(self.edge_pad_mask == 1, pt.tensor(1.0, device='cuda'), y_hat)
        y_hat = y_hat.squeeze()
        s_hat = self.radon_t.forward(y_hat)
        return y_hat, s_hat
    
class SinogramToPatchesConnectedReconstructor(ReconstructorBase):
    def __init__(self,
                 proj_dim: int,
                 angles,
                 patch_size=5,
                 stride=-1,
                 a = 0.1,
                 image_mask=None,
                 device="cuda",
                 edge_pad_size=0,
                 scale_sinogram=False):
        super().__init__(proj_dim,
                         angles,
                         a = a,
                         image_mask=image_mask,
                         device=device,
                         edge_pad_size=edge_pad_size)
        if scale_sinogram:
            raise NotImplementedError("Scaling sinogram not implemented for BackprojectAndUNet")
        if stride == -1:
            stride = patch_size
        self.nn = SinogramToPatchesConnected(proj_dim,patch_size,stride,angles)
        
    def forward(self, s):
        y_hat = self.nn(s)
        y_hat = pt.reshape(y_hat,(self.dim,self.dim))
        y_hat = self.image_mask * y_hat
        # Set all pixels that are in the edge_pad to 1
        y_hat = pt.where(self.edge_pad_mask == 1, pt.tensor(1.0, device='cuda'), y_hat)
        y_hat = y_hat.squeeze()
        s_hat = self.radon_t.forward(y_hat)
        return y_hat, s_hat
    
    def parameters(self):
        return self.nn.parameters()
    
class BackprojectAndUNet(ReconstructorBase):
    def __init__(self,
                 proj_dim: int,
                 angles, a = 0.1,
                 image_mask=None,
                 device="cuda",
                 edge_pad_size=0,
                 scale_sinogram=False):
        super().__init__(proj_dim,
                         angles,
                         a = a,
                         image_mask=image_mask,
                         device=device,
                         edge_pad_size=edge_pad_size)
        if scale_sinogram:
            raise NotImplementedError("Scaling sinogram not implemented for BackprojectAndUNet")
        self.nn = UNet(in_channels=1, out_channels=1, init_features=32)
        
    def forward(self, s):
        y_hat_prime = self.radon_t.backprojection(s)
        
        # Scale
        y_hat_prime = y_hat_prime.reshape((1,1,self.dim,self.dim))
        y_hat = self.nn(y_hat_prime)

        # Multiply elementwise with outer mask
        y_hat = y_hat * self.image_mask
        
        y_hat = pt.where(self.edge_pad_mask == 1, pt.tensor(1.0), y_hat)
        y_hat = y_hat.squeeze()
        s_hat = self.radon_t.forward(y_hat)
        return y_hat, s_hat
    
    def parameters(self):
        return self.nn.parameters()


def create_circle(radius=63, nholes=10, hole_volatility=0.4, n_missing_pixels=0.05, hole_ratio_limit=10):
    circle = Circle(radius)
    # Create holes in different angles
    for i in range(nholes):
        at_angle = np.random.randint(0,90)
        circle.make_holes(1,
                    n_missing_pixels,
                    hole_volatility,
                    hole_ratio_limit,
                    at_angle)
    return circle

def get_htc_scan(level = 1, sample = "a", return_raw_sinogram = False,angles=[]):
    base_path = "/home/ilmari/python/limited-angle-tomography/htc2022_test_data/"
    # htc2022_01a_recon_fbp_seg.png
    htc_file = f"htc2022_0{level}{sample}_recon_fbp_seg.png"
    sinogram_file = f"htc2022_0{level}{sample}_limited_sinogram.csv"
    angle_file = f"htc2022_0{level}{sample}_angles.csv"
    angle_file = os.path.join(base_path, angle_file)
    
    if not (isinstance(angles, np.ndarray) and angles.shape[0] > 0):
        # Load angles
        angles = np.loadtxt(angle_file,dtype=np.str_, delimiter=",")
        angles = np.array([float(angle) for angle in angles])

    # Load the img
    img = read_image(os.path.join(base_path, htc_file))
    img = pt.tensor(img, dtype=pt.float32, device='cuda', requires_grad=False)
    #img = img.squeeze()
    print(f"Sample {level}{sample} rotating to {angles[0]}")
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
        
        if level == 8:
            sample_to_degrees = {
                "a": 60,
                "b": 50,
                "c": 40,
                "d": 30,
            }
            # Only take a part of the sinogram
            angles = np.arange(0,sample_to_degrees[sample] + 0.5,0.5)
            sinogram = sinogram[0:len(angles),:]
        return y, outer_mask, sinogram, angles
    
    return y, outer_mask

def get_shepp_logan_scan(angles, image_dim = 512):
    phantom = ph.ct_shepp_logan(image_dim)
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

def load_base_images(path, to_tensor=True, to_3d=True):
    base_image_paths = list(filter(lambda x: "shape" in x, os.listdir(path)))
    base_image_paths = base_image_paths[0:10]
    to_tensor = True if to_3d else to_tensor
    # Load the numpy arrays
    base_images = []
    for image_path in base_image_paths:
        base_images.append(np.load(os.path.join(path, image_path)))
        
    if to_tensor:
        # Convert to torch tensors
        base_images = pt.tensor(base_images, dtype=pt.float32, device='cuda')
        print(f"Base images shape: {base_images.shape}")
    
    if to_3d:
        # Convert the base images to rgb
        base_images = base_images.unsqueeze(1)
        base_images = pt.cat([base_images, base_images, base_images], dim=1)
        print(f"Base images shape: {base_images.shape}")
    return base_images

def load_htc_images(path):
    base_image_paths = list(filter(lambda x: "recon" in x, os.listdir(path)))
    print(f"Loading images: {base_image_paths}")
    #base_image_paths = base_image_paths[0:10]
    # Load the numpy arrays
    base_images = []
    for image_path in base_image_paths:
        img = read_image(os.path.join(path, image_path), mode=ImageReadMode.GRAY)
        img = pt.tensor(img, dtype=pt.float32, device='cuda')
        img = img.squeeze()
        #print(f"Image shape: {img.shape}")
        base_images.append(img)
        
    return base_images

def find_good_a(s, sinogram, device='cuda',lims=(0,10), num_samples=200):
    diffs = []
    sinograms = []
    smallest_error = float("inf")
    smallest_error_a = 0
    for a in np.linspace(lims[0],lims[1],num_samples):
        if a == 0:
            sinogram_ = sinogram
        else:
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
    def __init__(self, p=1.0):
        """ Loss that computes the Lp norm of the difference between two images.
        """
        super(LPLoss, self).__init__()
        self.p = p
        
    def forward(self, X, Y):
        # calculate the mean of the Lp norm of the difference between X and Y
        return pt.mean(pt.abs(X - Y)**self.p * (1/self.p))
    
def scale_to_mean_and_std(X, mean, std):
    # Scale s_hat to have the same mean and std as s
    X_mean = pt.mean(X)
    X_std = pt.std(X)
    
    X_standardized = (X - X_mean) / X_std
    X_scaled = X_standardized * std + mean
    return X_scaled

def create_regularization(use_tv_reg = True,
                          use_bin_reg = False,
                          use_tik_reg = False,
                          use_autoencoder_reg = False,
                          autoencoder_path="",
                          autoencoder_patch_size=40,
                          autoencoder_latent_vars=10,
                          autoencoder_reconstruction_stride=5,
                          autoencoder_batch_size=128,
    ):
    regularizations = []
    if use_tv_reg:
        use_tv_reg = 1 if isinstance(use_tv_reg,bool) else use_tv_reg
        regularizations.append(lambda x : use_tv_reg * total_variation_regularization(x,normalize=True))
    if use_bin_reg:
        use_bin_reg = 1 if isinstance(use_bin_reg,bool) else use_bin_reg
        regularizations.append(lambda x : use_bin_reg * binary_regularization(x))
    if use_tik_reg:
        use_tik_reg = 1 if isinstance(use_tik_reg,bool) else use_tik_reg
        regularizations.append(lambda x : use_tik_reg * tikhonov_regularization(x,normalize=True))
    if use_autoencoder_reg:
        use_autoencoder_reg = 1 if isinstance(use_autoencoder_reg,bool) else use_autoencoder_reg
        regu = create_autoencoder_regularization(autoencoder_path,
                                                autoencoder_patch_size,
                                                autoencoder_latent_vars,
                                                autoencoder_reconstruction_stride,
                                                autoencoder_batch_size,
                                                )
        reg = lambda x : use_autoencoder_reg * regu(x)
        regularizations.append(reg)
        
    if not regularizations:
        return lambda x: pt.tensor(0.0, device=x.device)
    return lambda x: sum([r(x) for r in regularizations])
                          

def folder_name_from_params(base_name = "Benchmark", **kwargs):
    # Sort alphabetically
    kwargs = dict(sorted(kwargs.items()))
    for k,v in kwargs.items():
        base_name += "_" + str(k) + "=" + str(v)
    return base_name
    

def run_benchmark(**kwargs):
    do_levels = kwargs.get('do_levels', [5, 6,7])
    skip_done_levels = kwargs.get('skip_done_levels', False)
    filter_raw_sinogram_with_a = kwargs.get('filter_raw_sinogram_with_a', 5.0)
    filter_sinogram_of_predicted_image_with_a = kwargs.get('filter_sinogram_of_predicted_image_with_a', 5.0)
    p_loss = kwargs.get('p_loss', 1)
    scale_sinogram = kwargs.get('scale_sinogram', False)
    trim_sinogram = kwargs.get('trim_sinogram', False)
    pad_y_and_mask = kwargs.get('pad_y_and_mask', False)
    search_a_for_raw_sinogram = kwargs.get('search_a_for_raw_sinogram', False)
    scale_shat_to_same_mean_and_std = kwargs.get('scale_shat_to_same_mean_and_std', False)
    compare_s_score_to_unpadded = kwargs.get('compare_s_score_to_unpadded', False)
    use_otsu = kwargs.get("use_otsu", False)
    edge_pad_size = kwargs.get('edge_pad_size', 0)
    use_no_model = kwargs.get('use_no_model', False)
    sinogram_noise_std = kwargs.get('sinogram_noise_std', 0)
    time_limit_s = kwargs.get('time_limit_s', 60)
    use_tv_reg = kwargs.get('use_tv_reg', False)
    use_bin_reg = kwargs.get('use_bin_reg', False)
    use_tik_reg = kwargs.get('use_tik_reg', False)
    learning_rate = kwargs.get("learning_rate", 0.4 if use_no_model else 0.001)
    use_autoencoder_reg = kwargs.get('use_autoencoder_reg', False)
    autoencoder_path = kwargs.get('autoencoder_path', "")
    autoencoder_patch_size = kwargs.get('autoencoder_patch_size', 40)
    autoencoder_latent_vars = kwargs.get('autoencoder_latent_vars', 10)
    autoencoder_reconstruction_stride = kwargs.get('autoencoder_reconstruction_stride', 5)
    autoencoder_batch_size = kwargs.get('autoencoder_batch_size', 128)

    kwargs_for_naming = {
        "Model": not use_no_model,
        "P": p_loss,
        "Filt": filter_raw_sinogram_with_a,
        "Time": time_limit_s,
        "use_otsu": use_otsu,
    }

    if edge_pad_size != 0:
        kwargs_for_naming["EdgePad"] = edge_pad_size
    if use_tv_reg:
        kwargs_for_naming["TV"] = str(use_tv_reg)
    if use_bin_reg:
        kwargs_for_naming["BinReg"] = str(use_bin_reg)
    if use_tik_reg:
        kwargs_for_naming["TK"] = str(use_tik_reg)
    if use_autoencoder_reg:
        kwargs_for_naming["AutoEnc"] = f"Patch{autoencoder_patch_size}LV{autoencoder_latent_vars}"
        kwargs_for_naming["AutoEnc"] += f"Stride{autoencoder_reconstruction_stride}"
        kwargs_for_naming["Coeff"] = use_autoencoder_reg

    FOLDER = folder_name_from_params(base_name="NewBenchmark", **kwargs_for_naming)
    print(f"Folder: {FOLDER}")
    os.makedirs(FOLDER, exist_ok=True)
    
    # Write the parameters to a file
    with open(os.path.join(FOLDER, "params.json"), 'w') as f:
        json.dump(kwargs, f)

    regularization_ = create_regularization(
        use_tv_reg=use_tv_reg,
        use_bin_reg=use_bin_reg,
        use_tik_reg=use_tik_reg,
        use_autoencoder_reg=use_autoencoder_reg,
        autoencoder_path=autoencoder_path,
        autoencoder_patch_size=autoencoder_patch_size,
        autoencoder_latent_vars=autoencoder_latent_vars,
        autoencoder_reconstruction_stride=autoencoder_reconstruction_stride,
        autoencoder_batch_size=autoencoder_batch_size
    )

    criterion = LPLoss(p=p_loss)

    for htc_level in do_levels:
        samples = ["a", "b", "c"] if htc_level != 8 else ["a", "b", "c", "d"]
        level_performances = {s:[] for s in samples}
        for htc_sample in samples:
            print(f"Level {htc_level}, sample {htc_sample}")

            if skip_done_levels:
                summary_file = os.path.join(FOLDER, f"{htc_level}_summary.json")
                if os.path.exists(summary_file):
                    print(f"File {summary_file} exists, skipping level {htc_level}, sample {htc_sample}")
                    with open(summary_file, 'r') as f:
                        content = json.load(f)
                        if htc_sample in content.keys():
                            print(f"Skipping level {htc_level}, sample {htc_sample}")
                            continue

            loss_fig, loss_ax = plt.subplots(1, 3)
            loss_ax[0].set_title("Regularization loss")
            loss_ax[1].set_title("Reconstruction loss")
            loss_ax[2].set_title("MCC")
            loss_ax[0].set_xlabel("Iteration")
            loss_ax[1].set_xlabel("Iteration")
            loss_ax[2].set_xlabel("Iteration")
            loss_ax[0].set_ylabel("Regularization loss")
            loss_ax[1].set_ylabel(f"L{p_loss} loss")
            loss_ax[2].set_ylabel("MCC")

            regularization_losses = []
            reconstruction_losses = []
            mccs = []

            image_fig, image_ax = plt.subplots(2, 2)
            image_ax[0, 0].set_title("True image")
            image_ax[0, 1].set_title("Predicted image")
            image_ax[1, 0].set_title("True sinogram")
            image_ax[1, 1].set_title("Predicted sinogram")
            image_ax[1, 0].set_ylabel("Angle")
            image_ax[1, 1].set_ylabel("Angle")
            image_ax[1, 0].set_xlabel("Projection")
            image_ax[1, 1].set_xlabel("Projection")

            reconstruction_images = []
            
            try:
                y, image_mask, sinogram, angles = get_htc_scan(angles=[], level=htc_level, sample=htc_sample, return_raw_sinogram=True)
            except Exception as e:
                print(f"Error getting HTC scan: {e}")
                continue
            image_ax[1, 0].set_yticks(np.arange(0, len(angles), 10))
            image_ax[1, 1].set_yticks(np.arange(0, len(angles), 10))
            image_ax[1, 0].set_yticklabels(np.round(angles[::10], 2))
            image_ax[1, 1].set_yticklabels(np.round(angles[::10], 2))

            if sinogram is None:
                print(f"Sinogram is None, measuring sinogram")
                radon_t_no_filter = FBPRadon(y.shape[1], np.deg2rad(angles), a=0)
                sinogram = radon_t_no_filter.forward(y)
                mean = pt.mean(sinogram)
                std = pt.std(sinogram)
                noise = pt.normal(0.0, sinogram_noise_std, size=sinogram.shape)
                sinogram = sinogram + noise

            if image_mask is None:
                image_mask = pt.ones(y.shape, requires_grad=False)

            print(f"Shapes of data:\ny: {y.shape}, sinogram: {sinogram.shape}, image_mask: {image_mask.shape}, angles: {angles.shape}")

            assert not (trim_sinogram and pad_y_and_mask), "Cannot trim sinogram and pad y and mask"
            if trim_sinogram:
                num_extra_cols = sinogram.shape[1] - y.shape[1]
                if num_extra_cols != 0:
                    rm_from_left = num_extra_cols // 2 - 1
                    rm_from_right = num_extra_cols - rm_from_left
                    sinogram = sinogram[:, rm_from_left:sinogram.shape[1] - rm_from_right]
                    print(f"Trimmed sinogram to shape: {sinogram.shape}")

            if pad_y_and_mask:
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

            if use_no_model:
                model = NoModel(
                    proj_dim=sinogram.shape[1],
                    angles=np.deg2rad(angles),
                    a=filter_sinogram_of_predicted_image_with_a,
                    image_mask=image_mask,
                    edge_pad_size=edge_pad_size,
                )
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
            else:
                model = HTCModelReconstructor(
                    proj_dim=sinogram.shape[1],
                    angles=np.deg2rad(angles),
                    a=filter_sinogram_of_predicted_image_with_a,
                    image_mask=image_mask,
                    edge_pad_size=edge_pad_size,
                )
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

            model.to('cuda')

            raw_sinogram_mean = pt.mean(sinogram)
            raw_sinogram_std = pt.std(sinogram)
            raw_sinogram = sinogram

            if search_a_for_raw_sinogram:
                radon_t_no_filter = FBPRadon(y.shape[1], np.deg2rad(angles), a=filter_sinogram_of_predicted_image_with_a, device='cuda')
                computed_sinogram = radon_t_no_filter.forward(y)
                filter_raw_sinogram_with_a = find_good_a(computed_sinogram, raw_sinogram, device='cuda', lims=(0, 10), num_samples=200)
                print(f"The A filter value that when applied to the true sinogram, is closest to the computed sinogram is: {filter_raw_sinogram_with_a}")

            if filter_raw_sinogram_with_a != 0:
                filtered_sinogram = filter_sinogram(raw_sinogram, filter_raw_sinogram_with_a, device="cuda")
            else:
                filtered_sinogram = raw_sinogram
            if scale_sinogram:
                filtered_sinogram = (filtered_sinogram - pt.mean(filtered_sinogram)) / pt.std(filtered_sinogram)

            filtered_sinogram_mean = pt.mean(filtered_sinogram)
            filtered_sinogram_std = pt.std(filtered_sinogram)

            y_np = y.cpu().detach().numpy()
            filtered_sinogram_np = filtered_sinogram.cpu().detach().numpy()
            if compare_s_score_to_unpadded and pad_y_and_mask:
                y_not_padded_np = y_not_padded.cpu().detach().numpy()

            reconstruction_errors = []
            criterion_losses = []
            regularization_losses = []
            total_losses = []
            regularization_gradients = []
            loss_gradients = []
            regularization_grad_compared_to_loss_grad = []
            full_grad = pt.zeros(y.shape, device='cuda')
            criterion_loss_coeff = 1
            regularization_loss_coeff = 1

            iteration_number = 0
            start_time = time.time()
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > time_limit_s and len(reconstruction_images) > 0:
                    print(f"\nTime limit of {time_limit_s} seconds reached. Ending loop.")
                    break

                y_hat, s_hat = model(raw_sinogram)
                if iteration_number == 0:
                    try:
                        pass
                    except Exception as e:
                        pass
                y_hat = y_hat.reshape(y.shape)
                s_hat = s_hat.reshape((len(angles), y.shape[1]))

                if scale_shat_to_same_mean_and_std:
                    s_hat = scale_to_mean_and_std(s_hat, filtered_sinogram_mean, filtered_sinogram_std)

                criterion_loss = criterion(filtered_sinogram, s_hat)
                regularization_loss = regularization_(y_hat)
                y_hat_noise_reduced = y_hat
                loss = criterion_loss_coeff * criterion_loss + regularization_loss_coeff * regularization_loss

                criterion_losses.append(criterion_loss.item())
                regularization_losses.append(regularization_loss.item())
                total_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                y_hat_np = y_hat.cpu().detach().numpy()
                if use_otsu:
                    thresholds = threshold_multiotsu(y_hat_np, 2)
                    y_hat_np = np.digitize(y_hat_np, bins=thresholds).astype(np.uint8)
                    
                s_hat_np = s_hat.cpu().detach().numpy()
                M = np.zeros((2, 2))
                if compare_s_score_to_unpadded and pad_y_and_mask:
                    y_hat_np = y_hat_np[row_pad_up:-row_pad_down, col_pad_left:-col_pad_right]
                    y_np = y_not_padded_np

                y_hat_np = np.clip(y_hat_np, 0, 1)
                y_hat_rounded_np = np.round(y_hat_np)
                M[0, 0] = np.sum(np.logical_and(y_np == 1, y_hat_rounded_np == 1))
                M[0, 1] = np.sum(np.logical_and(y_np == 1, y_hat_rounded_np == 0))
                M[1, 0] = np.sum(np.logical_and(y_np == 0, y_hat_rounded_np == 1))
                M[1, 1] = np.sum(np.logical_and(y_np == 0, y_hat_rounded_np == 0))

                if len(np.unique(y_np)) == 2:
                    sc = (M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1]) / np.sqrt(
                        (M[0, 0] + M[0, 1]) * (M[1, 0] + M[1, 1]) * (M[0, 0] + M[1, 0]) * (M[0, 1] + M[1, 1]))
                else:
                    sc = ssim(y.unsqueeze(0).unsqueeze(0), y_hat.unsqueeze(0).unsqueeze(0), data_range=1)
                    sc = sc.item()

                reconstruction_errors.append(sc)
                iteration_number += 1
                regularization_losses.append(regularization_loss.item())
                reconstruction_losses.append(criterion_loss.item())
                mccs.append(sc)

                if iteration_number % 50 == 0 or iteration_number == 1:
                    reconstruction_images.append(y_hat_np)

                if iteration_number % 50 == 0:
                    s = f"Iteration: {iteration_number}, Loss: {loss.item()}, Criterion loss: {criterion_loss.item()}, Regularization loss: {regularization_loss.item()}, MCC: {sc}"
                    print(s, end="\r")
            print()
            loss_ax[0].plot(regularization_losses)
            loss_ax[1].plot(reconstruction_losses)
            loss_ax[2].plot(mccs)

            image_ax[0, 0].imshow(y_np)
            image_ax[0, 1].imshow(y_hat_np)
            image_ax[1, 0].imshow(filtered_sinogram_np)
            image_ax[1, 1].imshow(s_hat_np)

            loss_fig.savefig(f"{FOLDER}/{htc_level}{htc_sample}_loss.png")
            image_fig.savefig(f"{FOLDER}/{htc_level}{htc_sample}_images.png")

            reconstruction_images = [255 * x for x in reconstruction_images]
            imageio.mimsave(f"{FOLDER}/{htc_level}{htc_sample}_reconstruction.gif", reconstruction_images)

            level_performances[htc_sample] = {
                "regularization_losses": regularization_losses,
                "reconstruction_losses": reconstruction_losses,
                "mccs": mccs,
            }

        summary_file = f"{FOLDER}/{htc_level}_summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}
        
        print(level_performances)
        if level_performances["a"]:
            final_mccs = {sample: level_performances[sample]["mccs"][-1] for sample in level_performances.keys()}
            level_performances_with_final_mccs = {"final_mccs": final_mccs}
            level_performances_with_final_mccs.update(level_performances)
            existing_data.update(level_performances_with_final_mccs)
            with open(summary_file, "w") as f:
                json.dump(existing_data, f)
        else:
            print(f"No new results for level {htc_level}")

    levels_to_mccs = {}
    for level in do_levels:
        with open(f"{FOLDER}/{level}_summary.json", "r") as f:
            data = json.load(f)
            sample_mccs = data["final_mccs"]
            avg_mcc = np.sum([v for v in sample_mccs.values()])
            levels_to_mccs[level] = avg_mcc
    with open(f"{FOLDER}/summary.json", "w") as f:
        json.dump(levels_to_mccs, f)
    print(f"Results saved to {FOLDER}")
    
def plot_sinogram_and_filtered_sinogram(htc_level, htc_sample, a = 5.5  ):
    y, image_mask, sinogram, angles = get_htc_scan(angles=[], level=htc_level, sample=htc_sample, return_raw_sinogram=True)
    filtered_sinogram = filter_sinogram(sinogram, a=a, device='cuda')
    # Plot the unfiltered sinogram and the filtered sinogram
    fig, ax = plt.subplots(1,2)
    sinogram = filter_sinogram(sinogram, a=0.1, device='cuda')
    ax[0].imshow(sinogram.cpu().detach().numpy().T)
    ax[0].set_title("Sinogram filtered with a=0.1")
    ax[1].imshow(filtered_sinogram.cpu().detach().numpy().T)
    ax[1].set_title(f"Sinogram filtered with a={a}")
    plt.show()


def run_benchmark(
    do_levels=[5, 6, 7],
    skip_done_levels=False,
    filter_raw_sinogram_with_a=5.0,
    filter_sinogram_of_predicted_image_with_a=5.0,
    p_loss=1,
    scale_sinogram=False,
    trim_sinogram=False,
    pad_y_and_mask=False,
    search_a_for_raw_sinogram=False,
    scale_shat_to_same_mean_and_std=False,
    compare_s_score_to_unpadded=False,
    edge_pad_size=0,
    use_no_model=False,
    sinogram_noise_std=0,
    time_limit_s=60,
    use_tv_reg=False,
    use_bin_reg=False,
    use_tik_reg=False,
    use_autoencoder_reg=False,
    autoencoder_path="",
    autoencoder_patch_size=40,
    autoencoder_latent_vars=10,
    autoencoder_reconstruction_stride=5,
    autoencoder_batch_size=128,
    ):
    
    autoencoder_latent_vars = autoencoder_patch_size // 4
    autoencoder_reconstruction_stride = autoencoder_patch_size
    autoencoder_path = f"patch_autoencoder_P{autoencoder_patch_size}_D{autoencoder_latent_vars}.pth"
    if not os.path.exists(autoencoder_path):
        # Run ./learn_patch_autoencoder.py --patch_size x
        pyexe = sys.executable
        with open(f"patch_autoencoder_P{autoencoder_patch_size}_training_output.txt", "w") as f:
            result = subprocess.run([pyexe, "learn_patch_autoencoder.py", "--patch_size", str(autoencoder_patch_size)], stdout=f, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                print(f"Error training autoencoder with patch size {autoencoder_patch_size}. Check the output file for details.")
                exit(1)
    
    # Run ./learn_patch_autoencoder.py --patch_size x
    pyexe = sys.executable
    with open(f"patch_autoencoder_P{autoencoder_patch_size}_training_output.txt", "w") as f:
        result = subprocess.run([pyexe, "learn_patch_autoencoder.py", "--patch_size", str(autoencoder_patch_size)], stdout=f, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"Error training autoencoder with patch size {autoencoder_patch_size}. Check the output file for details.")
            exit(1)
    kwargs = {
        "do_levels": do_levels,
        "skip_done_levels": skip_done_levels,
        "filter_raw_sinogram_with_a": filter_raw_sinogram_with_a,
        "filter_sinogram_of_predicted_image_with_a": filter_sinogram_of_predicted_image_with_a,
        "p_loss": p_loss,
        "scale_sinogram": scale_sinogram,
        "trim_sinogram": trim_sinogram,
        "pad_y_and_mask": pad_y_and_mask,
        "search_a_for_raw_sinogram": search_a_for_raw_sinogram,
        "scale_shat_to_same_mean_and_std": scale_shat_to_same_mean_and_std,
        "compare_s_score_to_unpadded": compare_s_score_to_unpadded,
        "edge_pad_size": edge_pad_size,
        "use_no_model": use_no_model,
        "sinogram_noise_std": sinogram_noise_std,
        "time_limit_s": time_limit_s,
        "use_tv_reg": use_tv_reg,
        "use_otsu":False,
        "use_bin_reg": use_bin_reg,
        "use_tik_reg": use_tik_reg,
        "use_autoencoder_reg": use_autoencoder_reg,
        "autoencoder_path": autoencoder_path,
        "autoencoder_patch_size": autoencoder_patch_size,
        "autoencoder_latent_vars": autoencoder_latent_vars,
        "autoencoder_reconstruction_stride": autoencoder_reconstruction_stride,
        "autoencoder_batch_size": autoencoder_batch_size,
    }
    run_benchmark(**kwargs)
    
    

if __name__ == "__main__":
    pt.set_default_device('cuda')
    # Simple test
    if False:
        settings = {
            "do_levels": [5],
            "skip_done_levels": False,
            "filter_raw_sinogram_with_a": 5.0,
            "filter_sinogram_of_predicted_image_with_a": 5.0,
            "p_loss": 1,
            "scale_sinogram": False,
            "trim_sinogram": True,
            "pad_y_and_mask": False,
            "search_a_for_raw_sinogram": False,
            "use_otsu":True,
            "time_limit_s": 6,
            "use_no_model":True,
        }
        
        run_benchmark(**settings)
        
        
        
        exit()
    
    if False:
        # Fetch settings from best3["file_path"]
        best3 = pd.read_excel("best3.xlsx")
        settings_to_run = best3["file_path"].values
        
        # Run the benchmark with each settings
        for settings in settings_to_run:
            settings = settings + "_top3"
            params_file = f"{settings}/params.json"
            with open(params_file, 'r') as f:
                params = json.load(f)
            params["do_levels"] = list(range(7,9))
            params["use_otsu"] = True
            run_benchmark(**params)
            
        
        exit()
    do_levels=[5, 6, 7]
    skip_done_levels=False
    filter_raw_sinogram_with_a=5.0
    filter_sinogram_of_predicted_image_with_a=5.0
    p_loss=1
    scale_sinogram=False
    trim_sinogram=False
    pad_y_and_mask=False
    search_a_for_raw_sinogram=False
    scale_shat_to_same_mean_and_std=False
    compare_s_score_to_unpadded=False
    edge_pad_size=0
    use_no_model=False
    sinogram_noise_std=0
    time_limit_s=60
    use_tv_reg=False
    use_bin_reg=False
    use_tik_reg=False
    use_autoencoder_reg=False
    autoencoder_path=""
    autoencoder_patch_size=40
    autoencoder_latent_vars=10
    autoencoder_reconstruction_stride=5
    autoencoder_batch_size=128
    
    