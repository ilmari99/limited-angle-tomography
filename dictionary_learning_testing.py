import os
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from PIL import Image
from AbsorptionMatrices import Circle

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

def sparse_coding(images, dictionary, patch_size=8):
    """ Compute the sparse codes for the images
    """
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
    sparse_codes = np.dot(patches, dictionary.T)
    return sparse_codes

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

def get_basic_circle_image():
    circle = create_circle()
    y = circle.matrix
    y = np.pad(y, ((0,1),(0,1)))
    return y

def get_sparse_coeffs(img, dictionary, patch_size=8):
    """ Get the coefficients in Dict for ALL patches in the image
    """
    patches = extract_patches_2d(img, (patch_size, patch_size))
    patches = patches.reshape(patches.shape[0], -1)
    sparse_codes = np.dot(patches, dictionary.T)
    return sparse_codes

def remove_noise_from_image_dl(img, dictionary, patch_size=8):
    """ Remove noise from the image using the dictionary
    """
    sparse_codes = get_sparse_coeffs(img, dictionary, patch_size)
    reco = np.dot(sparse_codes, dictionary)
    reco = reco.reshape(-1, patch_size, patch_size)
    reco = reconstruct_from_patches_2d(reco, img.shape)
    return reco

def remove_noise_from_image_gaussian(img):
    """ Remove noise from the image using Gaussian filter
    """
    kernel = np.ones((3, 3)) / 9
    reco = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            patch = img[i-1:i+2, j-1:j+2]
            reco[i, j] = np.sum(patch * kernel)
    return reco

def get_htc_scan(level = 1, sample = "a"):
    base_path = "/home/ilmari/python/limited-angle-tomography/htc2022_test_data/"
    # htc2022_01a_recon_fbp_seg.png
    htc_file = f"htc2022_0{level}{sample}_recon_fbp_seg.png"
    #sinogram_file = f"htc2022_0{level}{sample}_limited_sinogram.csv"
    #angle_file = f"htc2022_0{level}{sample}_angles.csv"
    #angle_file = os.path.join(base_path, angle_file)
    
    print(f"Loading sample {level}{sample}",end="")

    # Read image
    img = Image.open(os.path.join(base_path, htc_file))
    img = np.array(img, dtype=np.float32)
    # Scale to [0, 1] and binarize
    #img = img / 255
    #img = np.where(img > 0.5, 1, 0)
    return img

def distort_image(img, area=16, shuffle_chance=0.5):
    """ Distorts the image by modifying areaxarea areas
    """
    img = img.copy()
    for i in range(0, img.shape[0], area):
        for j in range(0, img.shape[1], area):
            if np.random.rand() < shuffle_chance:
                area_i = min(area, img.shape[0] - i)
                area_j = min(area, img.shape[1] - j)
                area_pixels = img[i:i+area_i, j:j+area_j].ravel()
                g_distr = np.random.shuffle(area_pixels)
                g_distr = area_pixels.reshape(area_i, area_j)
                img[i:i+area_i, j:j+area_j] = g_distr
    # clip
    #img = np.clip(img, 0, 1)
    return img
    
if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    #images = [get_basic_circle_image() for _ in range(30)]
    images = [get_htc_scan(lev, sample) for lev in range(1,3) for sample in ["a", "b","c"]]
    images = np.array(images)
    print(f"Max of images: {images.max()}, Min of images: {images.min()}, Mean of images: {images.mean()}")
    print(f"Shape of images: {images.shape}")
    n_components = 8
    alpha = 0.1
    batch_size = 1
    patch_size = 8
    img_shape = images.shape[1:]

    dictionary = learn_dictionary(images, n_components, alpha, batch_size, patch_size)
    print(f"Shape of dictionary: {dictionary.shape}")
    sparse_codes = sparse_coding(images, dictionary,patch_size)
    print(f"Shape of encoding of all images: {sparse_codes.shape}")
    
    # Visualize the dict
    nrows = int(np.ceil(np.sqrt(n_components)))
    ncols = int(np.ceil(np.sqrt(n_components)))
    print(f"Number of rows: {nrows}")
    print(f"Number of columns: {ncols}")
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i in range(nrows):
        #print(f"i: {i}")
        for j in range(ncols):
            #print(f"j: {j}")
            idx = i * ncols + j
            if idx >= n_components:
                break
            #print(f"idx: {idx}")
            ax[i, j].imshow(dictionary[idx].reshape(patch_size, patch_size), cmap="gray")
            ax[i, j].axis("off")
    fig.suptitle("Learned dictionary")
    
    
    #test_imgs = [get_basic_circle_image() for _ in range(20)]
    test_imgs = [get_htc_scan(lvl, sample) for lvl in range(5,7) for sample in ["a", "b","c"]]
    test_imgs = np.array(test_imgs)
    
    print("Testing the fit of the dictionary on test images")
    reconstructions_with_dl = []
    reconstructions_with_gaussian = []
    for test_img in test_imgs:
        reco = remove_noise_from_image_dl(test_img, dictionary, patch_size)
        reco = (reco - reco.min()) / (reco.max() - reco.min())
        # Otsu
        #reco = np.where(reco > 0.5, 1, 0)
        reconstructions_with_dl.append(reco)
    reconstructions = np.array(reconstructions_with_dl)
    errors_with_dl = [np.linalg.norm(img - reco) for img, reco in zip(test_imgs, reconstructions_with_dl)]
    print(f"Mean reconstruction error: {np.mean(errors_with_dl)}")
    
    # Plot the reconstruction of the first test image
    test_img = test_imgs[-1]
    test_img = distort_image(test_img)
    # Corrupt the test image
    #test_img = test_img + 0.4 * np.random.randn(*test_img.shape)
    #test_img = np.clip(test_img, 0, 1)
    test_img = distort_image(test_img)
    sparse_codes = get_sparse_coeffs(test_img, dictionary, patch_size)
    reco = np.dot(sparse_codes, dictionary)
    reco = reco.reshape(-1, patch_size, patch_size)
    reco = reconstruct_from_patches_2d(reco, img_shape)
    # minmax scale
    reco = (reco - reco.min()) / (reco.max() - reco.min())
    reco = np.where(reco > 0.5, 1, 0)
    
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax[0].imshow(test_img, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Noisy image")
    ax[1].imshow(reco, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("Reconstructed image")
    ax[2].imshow(np.abs(test_img - reco), cmap="gray")
    ax[2].axis("off")
    ax[2].set_title("Difference")
    plt.show()