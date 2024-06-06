
from collections import OrderedDict
import os

import cv2
from AbsorptionMatrices import Circle

import torch as pt

import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt
from reconstruct import reconstruct_outer_shape
from utils import filter_sinogram
from utils import FBPRadon
from pytorch_msssim import ssim, ms_ssim
import phantominator as ph
from regularization import vector_similarity_regularization, number_of_edges_regularization, binary_regularization
from pytorch_models import SequenceToImageCNN, UNet, UNet2, SinogramCompletionTransformer, LSTMSinogram, HTCModel
from torchsummary import summary
import torchvision
import torchviz

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
    
class ReconstructFromSinogram(ModelBase):
    """ This class is for models, that predict the image directly from the sinogram.
    """
    def __init__(self, proj_dim: int, angles, a = 0.1, image_mask=None, device="cuda"):
        super().__init__(proj_dim, angles, a = a, image_mask=image_mask, device=device)
    
    def create_nn(self) -> nn.Module:
        htc_net = HTCModel((len(self.angles), self.dim), init_features=128, overwrite_cache=False, init_channels=1)#, load_weights="htc_trained_model.pth")
        htc_net.to("cuda")
        #summary(htc_net, (2,181,560), batch_size=1)
        return htc_net
    
    def forward(self, s):
        
        s = s.float()
        s.to(self.device)
        # Pad the sinogram to 181 x 512
        #missing_rows = 181 - s.shape[0]
        #s = pt.nn.functional.pad(s, (0,0,0,missing_rows))
        #s = s.reshape((1,1,181,self.dim))
        print(f"Sinogram shape: {s.shape}")
        #s = s.reshape((1,1,len(self.angles),self.dim))
        # Create a mask that is 1 where we have data, and 0 where we padded
        #mask = pt.ones((181,560), device='cuda', requires_grad=False)
        #mask[181-missing_rows:,:] = 0
        #s = s.reshape((1,1,181,560))
        #print(f"Sinogram shape: {s.shape}")
        #mask = mask.reshape((1,1,181,560))
        #print(f"Mask shape: {mask.shape}")
        
        # Concat mask on channel dimension
        #s = pt.cat([s, mask], dim=1)

        s = s.reshape((1,1,len(self.angles),self.dim))
        y_hat = self.nn(s)
        
        print(f"Y_hat shape: {y_hat.shape}")
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
        
        print(f"Yhat passed to radon_t: {y_hat.shape}")
        
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
        nn_ = UNet(in_channels=1, out_channels=1, init_features=32)#, load_weights="model_gpu.pt")
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
        # Standardize
        #sinogram = (sinogram - pt.mean(sinogram)) / pt.std(sinogram)
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
    def __init__(self, p=0.5):
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

HTC_LEVEL_TO_ANGLES = {
    7 : np.linspace(0, 30, 60, endpoint=False),
    6 : np.linspace(0, 40, 80, endpoint=False),
    5 : np.linspace(0, 50, 100, endpoint=False),
    4 : np.linspace(0, 60, 120, endpoint=False),
    3 : np.linspace(0, 70, 140, endpoint=False),
    2 : np.linspace(0, 80, 160, endpoint=False),
    1 : np.linspace(0, 90, 180, endpoint=False)
}

def regularization(y_hat):
    return pt.tensor(0.0, device='cuda', requires_grad=False)
    mat = number_of_edges_regularization(y_hat,coeff=1, filter_sz=3)
    # Return the mean of the matrix
    edge_regularization = pt.mean(mat)**2
    return edge_regularization
    
    y_hat = pt.cat([y_hat, y_hat, y_hat], dim=1)
    y_hat = y_hat.reshape((1,3,128,128))
    img_similarity = vector_similarity_regularization(y_hat, base_images, coeff=0.0001)
    img_similarity = pt.mean(img_similarity)
    return edge_regularization + img_similarity

if __name__ == "__main__":
    
    htc_level = 6
    htc_sample = "b"
    filter_raw_sinogram_with_a = 5.5
    filter_sinogram_of_predicted_image_with_a = 0.1
    criterion = LPLoss(p=1.0)
    trim_sinogram = True
    pad_y_and_mask = False
    search_a_for_raw_sinogram = False
    scale_shat_to_same_mean_and_std = False
    plot_rounded = False
    compare_s_score_to_unpadded = True
    
    angles = HTC_LEVEL_TO_ANGLES[htc_level]
    #angles = np.linspace(0,30,60, endpoint=False)
    sinogram = None
    #y, image_mask = get_basic_circle_scan(angles=angles)
    #y, image_mask = get_shepp_logan_scan(angles, image_dim=512)
    #y, image_mask = get_htc_scan(angles=angles, level=htc_level, sample=htc_sample, return_raw_sinogram=False)
    y, image_mask, sinogram, angles = get_htc_scan(angles=angles, level=htc_level, sample=htc_sample, return_raw_sinogram=True)
    
    if sinogram is None:
        print(f"Sinogram is None, measuring sinogram")
        radon_t_no_filter = FBPRadon(y.shape[1], np.deg2rad(angles), a=0, device='cuda')
        sinogram = radon_t_no_filter.forward(y)
    
    print(f"Shapes of data:\ny: {y.shape}, sinogram: {sinogram.shape}, image_mask: {image_mask.shape}, angles: {angles.shape}")

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
    

    
    model = ReconstructFromSinogram(sinogram.shape[1], np.deg2rad(angles), a=filter_sinogram_of_predicted_image_with_a, image_mask=image_mask, device='cuda')
    #model = PredictSinogramAndReconstruct(128, np.deg2rad(angles), image_mask=outer_mask, device='cuda')
    #model = BackprojectAndUNet(sinogram.shape[1], np.deg2rad(angles), a=A, image_mask=image_mask, device='cuda')
    model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    
    raw_sinogram_mean = pt.mean(sinogram)
    raw_sinogram_std = pt.std(sinogram)
    raw_sinogram = sinogram
    
    if search_a_for_raw_sinogram:
        # Find the value for A, that minimizes the difference between raw_sinogram,
        # and the sinogram computed from the true object.
        # This is not a realistic scenario, but we can use this to find a good quess for A during inference.
        radon_t_no_filter = FBPRadon(y.shape[1], np.deg2rad(angles), a=filter_sinogram_of_predicted_image_with_a, device='cuda')
        computed_sinogram = radon_t_no_filter.forward(y)
        filter_raw_sinogram_with_a = find_good_a(computed_sinogram, raw_sinogram, device='cuda', lims=(0,10), num_samples=200)
        print(f"The A filter value that when applied to the true sinogram, is closest to the computed sinogram is: {filter_raw_sinogram_with_a}")

    if filter_raw_sinogram_with_a != 0:
        filtered_sinogram = filter_sinogram(raw_sinogram,filter_raw_sinogram_with_a,device="cuda")
    else:
        filtered_sinogram = raw_sinogram
        
    filtered_sinogram_mean = pt.mean(filtered_sinogram)
    filtered_sinogram_std = pt.std(filtered_sinogram)
    
    print(f"Raw sinogram mean: {raw_sinogram_mean}")
    print(f"Raw sinogram std: {raw_sinogram_std}")
    print(f"Filtered sinogram mean: {filtered_sinogram_mean}")
    print(f"Filtered sinogram std: {filtered_sinogram_std}")

    # Plot the true y, naive reconstruction y_prime, and the NN predicted y
    image_fig, image_axes = plt.subplots(2,2, figsize=(10,5))
    # Plot the true y, and the predicted y
    y_np = y.cpu().detach().numpy()
    filtered_sinogram_np = filtered_sinogram.cpu().detach().numpy()
    if compare_s_score_to_unpadded and pad_y_and_mask:
        y_not_padded_np = y_not_padded.cpu().detach().numpy()
        image_axes[0][0].matshow(y_not_padded_np)
    else:
        image_axes[0][0].matshow(y_np)
    image_axes[0][1].matshow(filtered_sinogram_np)
    image_axes[0][0].set_title("True image")
    image_axes[0][1].set_title("Sinogram to optimize towards")
    image_axes[1][0].set_title("Predicted image")
    image_axes[1][1].set_title("Sinogram of predicted image")
    
    plt.show(block=False)

    loss_fig, loss_axes = plt.subplots(1,2, figsize=(10,5))
    loss_axes[0].set_title("Reconstruction error")
    loss_axes[0].set_xlabel("Iteration number")
    loss_axes[0].set_ylabel("Reconstruction error")
    loss_axes[1].set_title("Loss")
    loss_axes[1].set_xlabel("Iteration number")
    loss_axes[1].set_ylabel("Loss")

    reconstruction_errors = []
    criterion_losses = []
    regularization_losses = []
    total_losses = []
    
    iteration_number = 0
    while True:
        
        # The model returns the predicted image y_hat (y.shape)
        # and the predicted sinogram s (len(angles), y.shape[1])
        y_hat, s_hat = model(filtered_sinogram)
        y_hat = y_hat.reshape(y.shape)
        s_hat = s_hat.reshape((len(angles), y.shape[1]))# / 100
        
        # Scale s_hat to have the same mean and std as s
        if scale_shat_to_same_mean_and_std:
            s_hat = scale_to_mean_and_std(s_hat, filtered_sinogram_mean, filtered_sinogram_std)
            print(f"Scaled s_hat to mean: {pt.mean(s_hat)}, std: {pt.std(s_hat)}")
        
        # minimize criterion
        criterion_loss = criterion(filtered_sinogram,s_hat)
        
        regularization_loss = regularization(y_hat)
        
        loss = criterion_loss + regularization_loss
        
        criterion_losses.append(criterion_loss.item())
        regularization_losses.append(regularization_loss.item())
        total_losses.append(loss.item())
        
        # Update the model
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # Squeeze and detach
        y_hat_np = y_hat.cpu().detach().numpy()
        s_hat_np = s_hat.cpu().detach().numpy()
        print(f"y hat np shape {y_hat.shape}")
        # Let's use Matthews correlation coefficient between
        # Round y_hat, and calculate a confusion matrix
        # by comparing y_hat and y_np
        M = np.zeros((2,2))
        if compare_s_score_to_unpadded and pad_y_and_mask:
            print(f"Using original Y, and not the padded version")
            y_hat_np = y_hat_np[row_pad_up:-row_pad_down, col_pad_left:-col_pad_right]
            y_np = y_not_padded_np
            
        y_hat_np = np.clip(y_hat_np, 0, 1)
        y_hat_rounded_np = np.round(y_hat_np)

        # M = [[TP, FN], [FP, TN]]
        M[0,0] = np.sum(np.logical_and(y_np == 1, y_hat_rounded_np == 1))
        M[0,1] = np.sum(np.logical_and(y_np == 1, y_hat_rounded_np == 0))
        M[1,0] = np.sum(np.logical_and(y_np == 0, y_hat_rounded_np == 1))
        M[1,1] = np.sum(np.logical_and(y_np == 0, y_hat_rounded_np == 0))
        
        # MCC
        s_score = (M[0,0] * M[1,1] - M[1,0]*M[0,1]) / np.sqrt((M[0,0] + M[0,1]) * (M[1,0] + M[1,1]) * (M[0,0] + M[1,0]) * (M[0,1] + M[1,1]))
        reconstruction_errors.append(1 - s_score)
        
        iteration_number += 1
        if iteration_number == 1:
            # Generate compute graph
            #torchviz.make_dot(regularization_loss, params=dict(model.named_parameters())).render("compute_graph", format="png")
            # Show the first prediction
            y_hat_np_rounded = np.round(y_hat_np)
            if plot_rounded:
                image_axes[1][0].matshow(y_hat_np_rounded)
            else:
                image_axes[1][0].matshow(y_hat_np)
            image_axes[1][1].matshow(s_hat_np)
            image_fig.canvas.draw()
            image_fig.canvas.flush_events()
            print(f"Reconstruction error between y and y_hat: {reconstruction_errors[-1]}")
            input("Press enter to continue")
        
        elif iteration_number % 50 == 0:
            print(f"Reconstruction error between y and y_hat: {reconstruction_errors[-1]}")
            print(f"Loss between s and s_hat: {criterion_loss}")
            # Update the figure
            y_hat_np_rounded = np.round(y_hat_np)
            if plot_rounded:
                image_axes[1][0].matshow(y_hat_np_rounded)
            else:
                image_axes[1][0].matshow(y_hat_np)
            image_axes[1][1].matshow(s_hat_np)
            image_fig.canvas.draw()
            image_fig.canvas.flush_events()
            
            # Plot the loss and the reconstruction error
            loss_axes[0].plot(reconstruction_errors)
            loss_axes[1].plot(total_losses, color='black')
            loss_axes[1].plot(criterion_losses, color='red')
            loss_axes[1].plot(regularization_losses, color='blue')
            loss_axes[1].legend(["Total Loss", "Loss function", "Regularization"])
            loss_fig.canvas.draw()
            loss_fig.canvas.flush_events()
            plt.pause(0.01)
            #if reconstruction_errors[-1] < 0.001:
            #    break