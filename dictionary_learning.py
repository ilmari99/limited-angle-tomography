
# TESTING
from AbsorptionMatrices import Circle
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning

def extract_patches_2d_pt(image, patch_size=8):
    """ A Pytorch implementation of extract_patches_2d.
    It takes in an image (WxH) and returns all the (patch_sz x patch_sz) patches with stride=1.
    """
    # Add batch and channel dimensions
    image = image.unsqueeze(0).unsqueeze(0)
    patches = torch.nn.functional.unfold(image, patch_size, stride=1)
    patches = patches.permute(0, 2, 1).reshape(-1, patch_size, patch_size)
    return patches

def reconstruct_from_patches_2d_pt(patches, image_size):
    """ A Pytorch implementation of reconstruct_from_patches_2d.
    It takes in patches, and reconstructs the image, averaging overlapping patches
    """
    patch_size = patches.shape[-1]
    n_patches = patches.shape[0]
    # The number of patches per row and column is the image size minus the patch size plus 1
    n_patches_per_row = image_size[1] - patch_size + 1
    n_patches_per_col = image_size[0] - patch_size + 1

    image = torch.zeros(image_size)
    
    # We need to keep track of the number of patches that overlap each pixel
    counter_image = torch.zeros(image_size)
    # Iterate over the patches
    for i in range(n_patches):
        # Get the row index
        row = i // n_patches_per_row
        # Get the column index
        col = i % n_patches_per_row
        # Get the patch
        patch = patches[i]
        # Add the patch to the image
        image[row:row+patch_size, col:col+patch_size] += patch
        # Add the counter to the counter image
        counter_image[row:row+patch_size, col:col+patch_size] += 1
    # Divide the image by the counter image
    image = image / counter_image
    return image

def get_sparse_coeffs_pt(image, dictionary, patch_size=8):
    patches = extract_patches_2d_pt(image, patch_size)
    patches = patches.reshape(patches.shape[0], -1)
    sparse_codes = torch.mm(patches, dictionary.T)
    return sparse_codes

def remove_noise_from_image_dl_pt(image, dictionary, patch_size=8):
    sparse_codes = get_sparse_coeffs_pt(image, dictionary, patch_size)
    print(f"Sparse codes shape: {sparse_codes.shape}")
    # Reconstruct the image
    patches = torch.mm(sparse_codes, dictionary).reshape(-1, patch_size, patch_size)
    print(f"Patches shape: {patches.shape}")
    image_reco = reconstruct_from_patches_2d_pt(patches, image.shape)
    print(f"Reconstructed image shape: {image_reco.shape}")
    # Minmax scale
    image_reco = (image_reco - image_reco.min()) / (image_reco.max() - image_reco.min())
    # Otsu's thresholding
    thresh = torch.mean(image_reco)
    image_reco = torch.where(image_reco > thresh, torch.tensor(1.0), torch.tensor(0.0))
    return image_reco
        

def learn_dictionary(images, n_components, alpha, batch_size, patch_size=8):
    """ Learn a dictionary from the images
    """
    print(f"Images shape: {images.shape}")
    print(f"Patch size: {patch_size}")
    patches = []
    for img in images:
        img_patches = extract_patches_2d(img, (patch_size, patch_size))
        patches.append(img_patches)
    patches = np.array(patches)
    print(f"Extracted patches from {len(images)} images: {patches.shape}")
    # (n_images, n_patches_per_image, patch_size, patch_size)
    # Reshape the patches to (n_patches_per_image * n_images, patch_size * patch_size)
    patches = patches.reshape(patches.shape[0]*patches.shape[1], -1)
    print(f"Reshaped patches to: {patches.shape}")
    dictionary = MiniBatchDictionaryLearning(n_components=n_components,
                                             alpha=alpha,
                                             batch_size=batch_size)
    dictionary = dictionary.fit(patches)
    return dictionary.components_


#TESTING
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

def plot_dictionary(dictionary, patch_size=8):
    """ Plot the dictionary
    """
    n_components = dictionary.shape[0]
    n_rows = n_components // 4
    fig, ax = plt.subplots(n_rows, 4, figsize=(12, 12))
    for i in range(n_components):
        ax[i//4, i%4].imshow(dictionary[i].reshape(patch_size, patch_size), cmap='gray')
        ax[i//4, i%4].axis('off')
    return fig, ax

def get_basic_circle_image():
    circle = create_circle()
    y = circle.matrix
    y = np.pad(y, ((0,1),(0,1)))
    return y

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
    patch_size = 5
    # Create a basic circle image
    y = get_basic_circle_image()
    
    y_patches = extract_patches_2d_pt(torch.tensor(y), patch_size)
    print(f"Pytorch extracted patches shape: {y_patches.shape}")
    
    # Load images as numpy arrays, and use sklearn to learn the dictionary.
    images = [get_basic_circle_image() for _ in range(30)]
    images = np.array(images)
    basis = learn_dictionary(images, 4, 0.1, 1, patch_size)
    
    #_,_ = plot_dictionary(basis, patch_size)
    
    # When we have learned the dictionary, we convert it to a torch tensor, which we can use
    # to compute the sparse codes and remove noise from the image.
    basis = torch.tensor(basis)
    print(f"Basis shape: {basis.shape}")
    test_img = get_basic_circle_image()
    test_img_distorted = shuffle_local_pixels(test_img, area=16, shuffle_chance=0.3)
    # Remove noise from the image
    reco = remove_noise_from_image_dl_pt(torch.tensor(test_img_distorted), basis, patch_size)
    reco = reco.numpy()
    # Plot
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(test_img, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(test_img_distorted, cmap='gray')
    ax[1].set_title("Distorted Image")
    ax[2].imshow(reco, cmap='gray')
    ax[2].set_title("Reconstructed Image")
    
    plt.show()
    
    
    