import os
from matplotlib import pyplot as plt
import numpy as np
import torch as pt
from torchvision.io import read_image
from AbsorptionMatrices import Circle
from reconstruct import reconstruct_outer_shape
import phantominator
from kornia.filters import spatial_gradient
import torch

def total_variation_regularization(mat, normalize=True, order=1):
    """ Calculate the total variation of an image
    """
    # Calculate the gradient
    grad = spatial_gradient(mat.reshape(1,1,mat.shape[0],mat.shape[1]), order=order, mode="sobel")
    grad = grad.squeeze()
    # Check that there are no NaN values
    assert not torch.isnan(grad).any(), "There are NaN values in the gradient"
    grad_x = grad[0,:,:]
    grad_y = grad[1,:,:]
    # Anisotropic TV, since we need differentiability
    grad_x_abs = torch.abs(grad_x)
    grad_y_abs = torch.abs(grad_y)
    tv = torch.sum(grad_x_abs + grad_y_abs)
    if normalize:
        tv = tv / (mat.shape[0] * mat.shape[1])
    return tv

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
        #sinogram = pt.log(sinogram + 1)
        # Standardize
        #sinogram = (sinogram - pt.mean(sinogram)) / pt.std(sinogram)
        return y, outer_mask, sinogram, angles
    
    return y, outer_mask

def get_shepp_logan_scan(angles, image_dim = 512):
    phantom = phantominator.ct_shepp_logan(image_dim)
    y = pt.tensor(phantom, dtype=pt.float32, device='cuda')
    #sino = rt.forward(y)
    shape = Circle.from_matrix(phantom)
    _, dist_front, dist_back = shape.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
    outer_mask = reconstruct_outer_shape(angles, dist_front, dist_back, zero_threshold=0.1)
    outer_mask = pt.tensor(outer_mask, dtype=pt.float32, device='cuda', requires_grad=False)
    return y, outer_mask


def make_edge_filter(size=3):
    """ Make a filter that can be used to calculate the number of edges in an image.
    """
    # Only square filters are supported
    assert size % 2 == 1, "Size must be odd"
    filter_ = np.ones((size, size))
    # The middle element is size^2 - 1
    filter_[size//2, size//2] = size**2 - 1
    #print(f"Filter: {filter_}")
    filter_ = torch.tensor(filter_, device='cuda', dtype=torch.float32, requires_grad=True)
    return filter_

def number_of_edges_regularization(y_hat, filter_sz=3, coeff=0.1):
    """ If y_hat has many pixels that are not surrounded by similar pixels, penalize it.
    """
    # First, convert y_hat to a binary image
    #y_hat = y_hat > 0.5
    # Set all 0s to -1
    y_hat = y_hat * 2 - 1
    # Count the number of edges
    edge_filter = make_edge_filter(size=filter_sz)
    filter_sz = torch.tensor(filter_sz, device='cuda', dtype=torch.float32)
    # Convolve y_hat with the edge filter
    y_hat = y_hat.unsqueeze(0).unsqueeze(0)
    similarity_matrix = torch.nn.functional.conv2d(y_hat, edge_filter.unsqueeze(0).unsqueeze(0))
    similarity_matrix = similarity_matrix.squeeze()
    # Absolute value
    similarity_matrix = torch.abs(similarity_matrix)
    # Divide by (size^2 - 1)*2, to scale it to be between 0 and 1
    similarity_matrix = similarity_matrix / ((filter_sz**2 - 1)*2)
    similarity_matrix = 1-similarity_matrix
    # clip to 0 and 1
    similarity_matrix = torch.clip(similarity_matrix, 0, 1)
    #print(f"Max similarity: {torch.max(similarity_matrix)}")
    #print(f"Min similarity: {torch.min(similarity_matrix)}")
    #similarity_matrix = torch.tensor(similarity_matrix, device='cuda', dtype=torch.float32, requires_grad=True)
    return similarity_matrix

def detect_corners(y_hat, transpose_kernel=False):
    """ Find corners in the image.
    """
    corner_kernel = torch.tensor([[0,1,2],[-1,0,1],[-2,-1,0]], device='cuda', dtype=torch.float32)
    if transpose_kernel:
        corner_kernel = pt.flipud(corner_kernel)
    corner_kernel = corner_kernel.unsqueeze(0).unsqueeze(0)
    # Convolve y_hat with the corner filter
    y_hat = y_hat.unsqueeze(0).unsqueeze(0)
    corner_matrix = torch.nn.functional.conv2d(y_hat, corner_kernel, padding=1)
    corner_matrix = corner_matrix.squeeze()
    corner_matrix = torch.abs(corner_matrix)
    # Divide by 6, to scale it to be between 0 and 1
    corner_matrix = corner_matrix / pt.max(corner_matrix)
    # clip to 0 and 1
    corner_matrix = torch.clip(corner_matrix, 0, 1)
    return corner_matrix


def shuffle_local_pixels(image, area=16, shuffle_chance=0.5):
    """ Shuffle the pixels in a local area of the image
    """
    shuffled_image = image.copy()
    for i in range(0, image.shape[0], area):
        for j in range(0, image.shape[1], area):
            if np.random.rand() < shuffle_chance:
                # Shuffle the pixels in the torch image
                shuffled_image[i:i+area, j:j+area] = np.random.permutation(shuffled_image[i:i+area, j:j+area].ravel()).reshape(area, area)
    return shuffled_image

if __name__ == "__main__":
    # Test the htc scan
    y, outer_mask, sinogram, angles = get_htc_scan(level=7, sample="a", return_raw_sinogram=True)
    #y = shuffle_local_pixels(y.cpu().detach().numpy(), area=16, shuffle_chance=0.5)
    #y = torch.tensor(y, dtype=torch.float32, device='cuda')
    # Calculate the FFT of y, and plot the largest frequencies
    y_fft = torch.fft.fftn(y)
    print(f"Shape of y_fft: {y_fft.shape}, dtype: {y_fft.dtype}")
    # Calculate the magnitude
    y_fft_abs = torch.abs(y_fft)
    y_fft_abs_hist = y_fft_abs.cpu().detach().numpy().ravel()
    plt.hist(np.clip(y_fft_abs_hist, 0, 1000), bins=1000)
    plt.title("Histogram of FFT Magnitudes")
    # Calculate the log
    y_fft_abs = torch.log(y_fft_abs + 1)
    # Recreate the image using only the largest frequencies
    # Set the largest 10% of the frequencies to the original values
    y_fft_filt = y_fft * (y_fft_abs > torch.quantile(y_fft_abs, 0.9))
    # Reconstruct the image
    y_recon = torch.fft.ifftn(y_fft_filt)
    y_recon = torch.abs(y_recon)
    
    # Plot the 25 highest frequency waves
    fig, ax = plt.subplots(5, 5)
    # The total magnitude is the sum of the magnitudes of the x and y components
    total_magnitude = torch.sqrt(y_fft_abs[0]**2 + y_fft_abs[1]**2)
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            # Find the index of the idxth largest total magnitude
            idx_max = torch.argsort(total_magnitude, descending=True)[idx]
            print(f"Index of {idx}th largest magnitude: {idx_max}")
            # Find the x and y components of the frequency
            x_component = y_fft[0, idx_max]
            y_component = y_fft[1, idx_max]
            print(f"x component: {x_component}, y component: {y_component}")
            # Calculate the magnitude
            magnitude = torch.sqrt(x_component**2 + y_component**2)
            print(f"Magnitude: {magnitude}")
            # Calculate the time domain signal
            signal = torch.fft.ifftn(torch.stack([x_component, y_component]))
            signal = torch.abs(signal)
            ax[i, j].imshow(signal.cpu().detach().numpy(), cmap='gray')
            
            
            
    # Plot the image
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(y.cpu().detach().numpy(), cmap='gray')
    ax[0].set_title("Original Image")
    
    ax[1].imshow(y_fft_abs.cpu().detach().numpy(), cmap='gray')
    ax[1].set_title("FFT of Image")
    
    # Plot reconstruction
    ax[2].imshow(y_recon.cpu().detach().numpy(), cmap='gray')
    ax[2].set_title("Reconstructed Image")
    plt.show()#
    