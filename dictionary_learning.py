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
    
    #print(f"Patches shape: {patches.shape}")
    patches = patches.reshape(-1, patch_size*patch_size)
    patches = patches.unsqueeze(0).permute(0, 2, 1)
    #print(f"Patches shape: {patches.shape}")
    counter_image = torch.ones_like(patches, device=device)
    #counter_image = torch.nn.functional.unfold(counter_image.unsqueeze(0).unsqueeze(0), patch_size, stride=1)
    #print(f"Counter image shape: {counter_image.shape}")
    # Fold patches of ones to know how many patches overlap each pixel
    counter_image = torch.nn.functional.fold(counter_image, image_size, patch_size, stride=1)
    counter_image = counter_image.squeeze()
    #print(f"Counter image shape: {counter_image.shape}")
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
    #image = (image - image.min()) / (image.max() - image.min())
    #image = torch.where(image > 0.5, 1, 0)
    #return image
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
                                             positive_code=True,
                                             positive_dict=True,
                                             #tol=0.1,
    )
    # Filter out the patches that only have one value
    filtered_patches = []
    for patch in patches:
        if len(np.unique(patch)) > 1:
            filtered_patches.append(patch)
    filtered_patches = np.array(filtered_patches)
    dictionary = dictionary.fit(filtered_patches)
    return dictionary.components_

def learn_dictionary_custom(images, n_components, alpha, patch_size=8, num_iters=500):
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
    # Now, we want to minimize the following cost function:
    # ||X - D * Z||^2 + alpha * ||Z||_0
    # where X is the patches, D is the dictionary, and Z is the sparse codes.
    # We can solve this using gradient descent.
    # X has the patches (n_patches, patch_size * patch_size)
    X = torch.tensor(patches, dtype=torch.float32)
    # D is the dictionary that has the base components (n_components, patch_size * patch_size)
    D = torch.randn(n_components, patch_size * patch_size, dtype=torch.float32, requires_grad=True)
    # Initialize D as randomly selected True patches
    #D = torch.tensor(patches[np.random.choice(patches.shape[0], n_components, replace=False)], dtype=torch.float32, requires_grad=True)
    
    # Z has the sparse codes for each patch (n_patches, n_components)
    Z = torch.randn(X.shape[0], n_components, dtype=torch.float32, requires_grad=True)
    
    # We can now minimize the cost function using gradient descent
    # We will use the Adam optimizer
    optimizer = torch.optim.Adam([D, Z], lr=0.1)
    # We will use the L2 norm as the reconstruction term
    l2_norm = torch.nn.MSELoss()
    
    # Split the patches to training and validation sets
    X_train = X
    for itern in range(num_iters):
        optimizer.zero_grad()
        # Compute the cost function
        loss = l2_norm(X_train, torch.mm(Z, D))
        reg = torch.sum(torch.abs(Z)) / (Z.shape[0]*Z.shape[1])
        print(f"Epoch {itern}, mse: {loss}, sparsity: {reg}")
        cost = loss + alpha * reg
        cost.backward()
        optimizer.step()
        
        if itern % 10 == 0:
            print(f"Epoch {itern}, cost: {cost}")
    return D.detach().cpu().numpy().astype(np.float32)

    
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
                min_i_idx = i
                max_i_idx = min(i + area, image.shape[0])
                min_j_idx = j
                max_j_idx = min(j + area, image.shape[1])
                # Get the pixels
                pixels = shuffled_image[min_i_idx:max_i_idx, min_j_idx:max_j_idx]
                # Shuffle the pixels
                pixels = pixels.flatten()
                np.random.shuffle(pixels)
                pixels = pixels.reshape(max_i_idx-min_i_idx, max_j_idx-min_j_idx)
                shuffled_image[min_i_idx:max_i_idx, min_j_idx:max_j_idx] = pixels
                
    return shuffled_image

def load_htc_images(path,device=None):
    base_image_paths = list(filter(lambda x: "recon" in x, os.listdir(path)))
    #base_image_paths = base_image_paths[0:10]
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
    
    patch_size = 16
    num_components = 9
    dl_alpha = 10
    # Load images as numpy arrays, and use sklearn to learn the dictionary.
    #images = [get_basic_circle_image() for _ in range(10)]
    images = load_htc_images("HTC_files")
    if images[0].device.type == 'cuda':
        images = [img.cpu().numpy() for img in images]
    images = np.array(images)
    basis = learn_dictionary(images, num_components, dl_alpha, patch_size)
    
    _,_ = plot_dictionary(basis, patch_size)
    #plt.show()
    #exit()
    
    # When we have learned the dictionary, we convert it to a torch tensor, which we can use
    # to compute the sparse codes and remove noise from the image.
    basis = torch.tensor(basis)
    print(f"Basis shape: {basis.shape}")

    test_img = get_htc_scan(7, "b")
    
    test_patches = extract_patches_2d_pt(torch.tensor(test_img,device="cpu"), patch_size, device="cpu")
    #Shuffle patches
    test_patches = test_patches.cpu().numpy()
    test_patches = test_patches[np.random.permutation(test_patches.shape[0])]
    # Plot 9 patches
    patch_fig, patch_ax = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            patch_ax[i, j].imshow(test_patches[idx].squeeze(), cmap='gray')
            patch_ax[i, j].axis('off')
    #plt.show()
    #exit()
    
    test_img_distorted = shuffle_local_pixels(test_img, area=16, shuffle_chance=0.5)
    # Remove noise from the image
    reco = remove_noise_from_image_dl_pt(torch.tensor(test_img_distorted), basis, patch_size)
    if reco.device.type == 'cuda':
        reco = reco.cpu()
    reco = reco.numpy()
    
    # Select oner andom patch, and reconstruct it using the dictionary
    patch = test_patches[0]
    while len(np.unique(patch)) < 2:
        patch = test_patches[np.random.randint(0, test_patches.shape[0])]
        
    patch = torch.tensor(patch, device="cuda")
    #patch = patch.unsqueeze(0)
    #patch = patch.unsqueeze(0)
    print(f"Patch shape: {patch.shape}")
    sparse_codes = get_sparse_coeffs_pt(patch, basis, patch_size)
    
    # Reconstruct the patch
    reco_patch = torch.mm(sparse_codes, basis)
    reco_patch = reco_patch.reshape(-1, patch_size, patch_size)
    reco_patch = reco_patch.squeeze().squeeze()
    reco_patch = reco_patch.cpu().numpy()
    # minmax scale
    #reco_patch = (reco_patch - reco_patch.min()) / (reco_patch.max() - reco_patch.min())
    # Plot the patch and the reconstructed patch
    patch_reco_fig, patch_reco_ax = plt.subplots(1, 2)
    patch_reco_ax[0].imshow(patch.squeeze().squeeze().cpu().numpy(), cmap='gray')
    patch_reco_ax[0].set_title("Original Patch")
    patch_reco_ax[1].imshow(reco_patch, cmap='gray')
    patch_reco_ax[1].set_title("Reconstructed Patch")
    patch_reco_fig.suptitle("Coeffs: " + str(sparse_codes))
    #plt.show()
    
    l1_norm_dist = np.sum(np.abs(test_img - test_img_distorted))
    print(f"L1 norm between original and distorted: {l1_norm_dist}")
    l2_norm_dist = np.sum((test_img - test_img_distorted) ** 2)
    print(f"L2 norm between original and distorted: {l2_norm_dist}")
    
    l1_norm = np.sum(np.abs(test_img - reco))
    print(f"L1 norm between original and reconstructed: {l1_norm}")
    l2_norm = np.sum((test_img - reco) ** 2)
    print(f"L2 norm between original and reconstructed: {l2_norm}")
    
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(test_img, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(test_img_distorted, cmap='gray')
    ax[1].set_title("Distorted Image")
    ax[2].imshow(reco, cmap='gray')
    ax[2].set_title("Reconstructed Image")
    
    plt.show()
    
    
    