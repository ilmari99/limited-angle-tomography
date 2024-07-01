import os
import torch
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning, DictionaryLearning
from torchvision.transforms.functional import adjust_sharpness
from torchvision.io import read_image, ImageReadMode
from PIL import Image

# TESTING
from AbsorptionMatrices import Circle
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt

def extract_patches_2d_pt(image, patch_size=8,device=None):
    """ A Pytorch implementation of extract_patches_2d.
    It takes in an image (WxH) and returns all the (patch_sz x patch_sz) patches with stride=1.
    """
    # Add batch and channel dimensions
    image = image.unsqueeze(0).unsqueeze(0)
    patches = torch.nn.functional.unfold(image, patch_size, stride=1)
    patches = patches.permute(0, 2, 1).reshape(-1, patch_size, patch_size)
    patches = patches.to(device)
    return patches

def reconstruct_from_patches_2d_pt_old(patches, image_size, device=None):
    """ A Pytorch implementation of reconstruct_from_patches_2d.
    It takes in patches, and reconstructs the image, averaging overlapping patches
    """
    patch_size = patches.shape[-1]
    n_patches = patches.shape[0]
    # The number of patches per row and column is the image size minus the patch size plus 1
    n_patches_per_row = image_size[1] - patch_size + 1
    n_patches_per_col = image_size[0] - patch_size + 1

    image = torch.zeros(image_size, device=device)
    
    # We need to keep track of the number of patches that overlap each pixel
    counter_image = torch.zeros(image_size, device=device)
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
    image = image.to(device)
    return image

def reconstruct_from_patches_2d_pt(patches, image_size, device=None):
    """ A Pytorch implementation of reconstruct_from_patches_2d.
    It takes in patches, and reconstructs the image, averaging overlapping patches
    """
    patch_size = patches.shape[-1]
    
    print(f"Patches shape: {patches.shape}")
    patches = patches.reshape(-1, patch_size*patch_size)
    patches = patches.unsqueeze(0).permute(0, 2, 1)
    print(f"Patches shape: {patches.shape}")
    counter_image = torch.ones_like(patches, device=device)
    #counter_image = torch.nn.functional.unfold(counter_image.unsqueeze(0).unsqueeze(0), patch_size, stride=1)
    print(f"Counter image shape: {counter_image.shape}")
    # Fold patches of ones to know how many patches overlap each pixel
    counter_image = torch.nn.functional.fold(counter_image, image_size, patch_size, stride=1)
    counter_image = counter_image.squeeze()
    print(f"Counter image shape: {counter_image.shape}")
    # fold the patches
    image = torch.nn.functional.fold(patches, image_size, patch_size, stride=1)
    image = image.squeeze()
    # Divide the image by the counter image
    image = image / counter_image
    image = image.to(device)
    return image
    

def get_sparse_coeffs_pt(image, dictionary, patch_size=8, device=None):
    patches = extract_patches_2d_pt(image, patch_size, device)
    patches = patches.reshape(patches.shape[0], -1)
    sparse_codes = torch.mm(patches, dictionary.T)
    return sparse_codes

def remove_noise_from_image_dl_pt(image, dictionary, patch_size=8,device=None):
    sparse_codes = get_sparse_coeffs_pt(image, dictionary, patch_size,device=device)
    #print(f"Sparse codes shape: {sparse_codes.shape}")
    # Reconstruct the image
    patches = torch.mm(sparse_codes, dictionary).reshape(-1, patch_size, patch_size)
    #print(f"Patches shape: {patches.shape}")
    image_reco = reconstruct_from_patches_2d_pt(patches, image.shape,device=device)
    #print(f"Reconstructed image shape: {image_reco.shape}")
    # Sharpen the image
    #image_reco = adjust_sharpness(image_reco.unsqueeze(0).unsqueeze(0), 2.0)
    #image_reco = image_reco.squeeze().squeeze()
    # Minmax scale
    image_reco = (image_reco - image_reco.min()) / (image_reco.max() - image_reco.min())
    # Otsu's thresholding
    image_reco = torch.where(image_reco > 0.5, 1, 0)
    return image_reco

def learn_dictionary(images, n_components, alpha, patch_size=8):
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
                                             max_iter=10000,
                                             transform_max_iter=10000,
                                             transform_algorithm="omp",
                                             transform_alpha=alpha,
                                             transform_n_nonzero_coefs=1,
                                             n_jobs=8,
                                             fit_algorithm="cd",
                                             #positive_code=True,
                                             positive_dict=True,
                                             #tol=0.1,
    )
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
    nrows = int(np.ceil(np.sqrt(n_components)))
    ncols = int(np.ceil(np.sqrt(n_components)))
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx < n_components:
                ax[i, j].imshow(dictionary[idx].reshape(patch_size, patch_size), cmap='gray')
                ax[i, j].axis('off')
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

def load_htc_images(path,device=None):
    base_image_paths = list(filter(lambda x: "recon" in x, os.listdir(path)))
    base_image_paths = base_image_paths[0:10]
    # Load the numpy arrays
    base_images = []
    for image_path in base_image_paths:
        img = read_image(os.path.join(path, image_path), mode=ImageReadMode.GRAY)
        img = torch.tensor(img, dtype=torch.float32, device=device)
        img = img.squeeze() / 255
        #print(f"Image shape: {img.shape}")
        base_images.append(img)
    return base_images

def get_htc_scan(level = 1, sample = "a"):
    base_path = "/home/ilmari/python/limited-angle-tomography/htc2022_test_data/"
    # htc2022_01a_recon_fbp_seg.png
    htc_file = f"htc2022_0{level}{sample}_recon_fbp_seg.png"
    #sinogram_file = f"htc2022_0{level}{sample}_limited_sinogram.csv"
    #angle_file = f"htc2022_0{level}{sample}_angles.csv"
    #angle_file = os.path.join(base_path, angle_file)
    
    print(f"Loading sample {level}{sample}")

    # Read image
    img = Image.open(os.path.join(base_path, htc_file))
    img = np.array(img, dtype=np.float32)
    # Scale to [0, 1] and binarize
    max_val = np.max(img)
    img = img / max_val
    #img = np.where(img > 0.5, 1, 0)
    return img

if __name__ == "__main__":

    torch.set_default_device('cuda')
    
    patch_sizes = [4, 6, 8, 10, 12, 14]
    num_components = [4, 6, 8, 10, 12, 14]
    dl_alpha = [0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.8, 4.4, 5.0]
    result_dict = {}
    
    images = load_htc_images("HTC_files")
    if images[0].device.type == 'cuda':
        images = [img.cpu().numpy() for img in images]
    images = np.array(images)
    for alpha in dl_alpha:
        for patch_size in patch_sizes:
            for n_components in num_components:
                print(f"Patch size: {patch_size}, Number of components: {n_components}")
                
                basis = learn_dictionary(images, n_components, alpha, patch_size)
                
                _,_ = plot_dictionary(basis, patch_size)
                #plt.show()
                #exit()
                
                # When we have learned the dictionary, we convert it to a torch tensor, which we can use
                # to compute the sparse codes and remove noise from the image.
                basis = torch.tensor(basis)
                print(f"Basis shape: {basis.shape}")
            
                test_img = get_htc_scan(7, "a")
                test_img_distorted = shuffle_local_pixels(test_img, area=32, shuffle_chance=0.4)
                # Remove noise from the image
                reco = remove_noise_from_image_dl_pt(torch.tensor(test_img_distorted), basis, patch_size)
                if reco.device.type == 'cuda':
                    reco = reco.cpu()
                reco = reco.numpy()
                
                l1_norm_dist = np.sum(np.abs(test_img - test_img_distorted))
                print(f"L1 norm between original and distorted: {l1_norm_dist}")
                l2_norm_dist = np.sum((test_img - test_img_distorted) ** 2)
                print(f"L2 norm between original and distorted: {l2_norm_dist}")
                
                l1_norm = np.sum(np.abs(test_img - reco))
                print(f"L1 norm between original and reconstructed: {l1_norm}")
                l2_norm = np.sum((test_img - reco) ** 2)
                print(f"L2 norm between original and reconstructed: {l2_norm}")
                result_dict[(patch_size, n_components,alpha)] = (l1_norm, l2_norm, l1_norm_dist, l2_norm_dist)
    print(f"Results: {result_dict}")
    # Sort the dict based on the L1 norm
    sorted_results = sorted(result_dict.items(), key=lambda x: x[1][0])
    print(f"Sorted results: {sorted_results}")
    best_params = sorted_results[0][0]
    print(f"Best parameters: {best_params}")
    
    
    
    