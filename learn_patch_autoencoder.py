import os
import random
import torch
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning, DictionaryLearning
from torchvision.transforms.functional import adjust_sharpness
from torchvision.io import read_image, ImageReadMode
from PIL import Image

# TESTING
from AbsorptionMatrices import Circle
from sklearn.feature_extraction.image import extract_patches_2d
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def extract_patches_2d_pt(image, patch_size=8,device=None, dtype=None, stride=1):
    """ A Pytorch implementation of extract_patches_2d.
    It takes in an image (WxH) and returns all the (patch_sz x patch_sz) patches with stride=1.
    """
    # Add batch and channel dimensions
    image = image.unsqueeze(0).unsqueeze(0)
    patches = torch.nn.functional.unfold(image, patch_size, stride=stride)
    patches = patches.permute(0, 2, 1).reshape(-1, patch_size, patch_size)
    patches = patches.to(device, dtype=dtype)
    return patches

def reconstruct_from_patches_2d_pt(patches, image_size, device=None, stride=1):
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
    counter_image = torch.nn.functional.fold(counter_image, image_size, patch_size, stride=stride)
    counter_image = counter_image.squeeze()
    #print(f"Counter image shape: {counter_image.shape}")
    # fold the patches
    image = torch.nn.functional.fold(patches, image_size, patch_size, stride=stride)
    image = image.squeeze()
    # Divide the image by the counter image
    image = image / counter_image
    image = image.to(device)
    return image

class PatchAutoencoder(torch.nn.Module):
    def __init__(self, patch_size = 16, num_latent = 8, pretrained_weights=None):
        super(PatchAutoencoder, self).__init__()
        self.encoder, self.decoder = self.get_autoencoder(patch_size, num_latent)
        if pretrained_weights:
            self.load_state_dict(torch.load(pretrained_weights))
    
    def encode(self, x):
        x = self.encoder(x)
        return x
        
    def forward(self, x):
        # Expand
        x = x.unsqueeze(1)
        # Encode
        x = self.encode(x)
        # Decode
        x = self.decoder(x)
        return x
    
    @staticmethod
    def get_autoencoder(patch_size = 16, num_latent = 8):
        """ Create an autoencoder model that learns to represent different types of patches.
        """
        # Encode the patch to num_latent values
        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64*patch_size*patch_size, num_latent),
        )
        decoder = torch.nn.Sequential(
            torch.nn.Linear(num_latent, 64*patch_size*patch_size),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (64, patch_size, patch_size)),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )
        return encoder, decoder
    
    def remove_noise_from_img_differentiable(self, img, patch_size,stride=-1):
        """ Remove noise from an image using the autoencoder.
        """
        if stride == -1:
            stride = patch_size
        patches = extract_patches_2d_pt(img, patch_size, stride=stride)
        enc_dec = self(patches)
        reconstructed = reconstruct_from_patches_2d_pt(enc_dec, img.shape, stride=stride)
        return reconstructed
    
    def remove_noise_from_img(self, img, patch_size, stride, batch_size, patches_to_device='cuda', patches_to_dtype=torch.float32):
        """ Remove noise from an image using the autoencoder.
        """
        # Extract patches
        patches = extract_patches_2d_pt(img, patch_size, stride=stride, device=patches_to_device, dtype=patches_to_dtype)
        batch_size = batch_size if batch_size > 0 else len(patches)
        dec_patches = []
        # Encode in batches
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            if patches_to_device != "cuda":
                batch = batch.to("cuda")
            if patches_to_dtype != torch.float32:
                batch = batch.to(torch.float32)
            dec = self(batch)
            dec_patches.append(dec)
        dec_patches = torch.cat(dec_patches, dim=0)
        dec_patches = dec_patches.squeeze(1)
        # Reconstruct the image
        reconstructed = reconstruct_from_patches_2d_pt(dec_patches, img.shape, stride=stride)
        return reconstructed

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

def random_distort(image, factor=0.5):
    """ Smooth the pixels of the image by resizing it and then resizing it back.
    """
    og_shape = image.shape
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0).unsqueeze(0)
    image = torch.nn.functional.interpolate(image, scale_factor=factor, mode='bilinear')
    image = torch.nn.functional.interpolate(image, mode='bilinear',size=og_shape)
    if image.device.type == 'cuda':
        image = image.cpu()
    image = image.squeeze().numpy()
    return image

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

def load_generated_images(path,device=None):
    images = []
    for image_path in os.listdir(path):
        if image_path.endswith(".png"):
            img = read_image(os.path.join(path, image_path), mode=ImageReadMode.GRAY)
            img = torch.tensor(img, dtype=torch.float32, device=device)
            img = img.squeeze() / 255
            # Round to 0 or 1
            img = torch.round(img)
            images.append(img)
    return images

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


def training_loop(autoenc,
                  train_filtered_patches,
                  test_filtered_patches,
                  num_epochs=5,
                  patience=10,
                  learning_rate=0.001,
                  batch_size=64,
                  restore_best=True):
    # Train the autoencoder
    optimizer = torch.optim.Adam(autoenc.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    # Move to device
    autoenc = autoenc.to(torch.get_default_device())
    #train_filtered_patches = train_filtered_patches.to(torch.get_default_device())
    #test_filtered_patches = test_filtered_patches.to(torch.get_default_device())
    
    
    # Train the autoencoder
    best_loss = np.inf
    best_epoch = 0
    best_weights = None
    for epoch in range(num_epochs):
        train_filtered_patches = train_filtered_patches[torch.randperm(len(train_filtered_patches),device="cpu")]
        for i in range(0, len(train_filtered_patches), batch_size):
            batch = train_filtered_patches[i:i+batch_size]
            batch = torch.tensor(batch, dtype=torch.float32).to(torch.get_default_device())
            optimizer.zero_grad()
            decoded = autoenc(batch)
            decoded = decoded.squeeze(1)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
        # Test the model
        with torch.no_grad():
            for i in range(0, len(test_filtered_patches), batch_size):
                batch = test_filtered_patches[i:i+batch_size]
                batch = torch.tensor(batch, dtype=torch.float32).to(torch.get_default_device())
                decoded = autoenc(batch)
                decoded = decoded.squeeze(1)
                test_loss = criterion(decoded, batch)
            print(f"Epoch {epoch}, Loss: {test_loss}")
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch
                best_weights = autoenc.state_dict()
            if epoch - best_epoch > patience:
                break
    # Restore the best weights
    if restore_best:
        autoenc.load_state_dict(best_weights)
    # Save the model
    torch.save(autoenc.state_dict(), "patch_autoencoder.pth")
    
    return autoenc


if __name__ == "__main__":
    patch_size = 40
    num_latent = 5
    num_epochs = 25
    patience = 5
    restore_best = True
    learning_rate = 0.001
    batch_size = 64
    train_test_split = 0.7
    load_pre_trained = "patch_autoencoder_P40_D5_also_synth.pth"
    do_training = False
    stride = 4
    
    torch.manual_seed(0)
    
    
    # Check that GPU is available
    if torch.cuda.is_available():
        print("Using GPU")
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')
    
    # Load images as numpy arrays, and use sklearn to learn the dictionary.
    #images = [get_basic_circle_image() for _ in range(10)]
    images = load_generated_images("generated_data")    
    if images[0].device.type == 'cuda':
        images = [img.cpu().numpy() for img in images]
    if not do_training:
        images = random.choices(images, k = 3)
    print(f"Loading {len(images)} images")
    images = np.array(images)
    
    images = [torch.tensor(img, dtype=torch.float32) for img in images]
    image_patches = [extract_patches_2d_pt(img, patch_size,"cuda",dtype=torch.bool,stride=stride) for img in images]
    
    # Concatenate the patches
    all_patches = torch.cat(image_patches, dim=0)
    all_patches = all_patches.to("cpu")
    print(f"All patches shape: {all_patches.shape}")
    
    # Filter out the patches where > 80% of the pixels are zero
    patch_sum = all_patches.sum(dim=(1,2), dtype=torch.float16)
    patch_sum = patch_sum / (patch_size * patch_size)
    filtered_patches = all_patches#[patch_sum > 0.2]
    print(f"Filtered patches shape: {filtered_patches.shape}")
    
    #filtered_patches = filtered_patches.to(torch.get_default_device())
    
    # Split the patches to train and test
    random_indices = torch.randperm(len(filtered_patches), device="cpu")
    train_indices = random_indices[0:int(train_test_split*len(filtered_patches))]
    train_filtered_patches = filtered_patches[train_indices]
    test_filtered_patches = filtered_patches[random_indices[int(train_test_split*len(filtered_patches)):]]
    
    autoenc = PatchAutoencoder(patch_size, num_latent)
    
    autoenc = autoenc.to(torch.get_default_device())
    #train_filtered_patches = train_filtered_patches.to(torch.get_default_device())
    #test_filtered_patches = test_filtered_patches.to(torch.get_default_device())
    
    if load_pre_trained:
        autoenc.load_state_dict(torch.load(load_pre_trained))
        
    if do_training:
        autoenc = training_loop(autoenc,
                                train_filtered_patches,
                                test_filtered_patches,
                                num_epochs=num_epochs,
                                patience=patience,
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                restore_best=restore_best)
    
    
    # Show 4 enc-dec examples
    fig, ax = plt.subplots(2, 4)
    fig.suptitle("Encoded and decoded patches")
    with torch.no_grad():
        for i in range(4):
            idx = np.random.randint(0, len(test_filtered_patches))
            original = test_filtered_patches[idx].clone().squeeze().cpu().numpy()
            original = original.astype(np.float32)
            original_distorted = shuffle_local_pixels(original, area=patch_size // 4, shuffle_chance=0.3)
            original_distorted = random_distort(original_distorted)
            ax[0,i].matshow(original)
            
            enc = autoenc.encode(torch.tensor(original_distorted).unsqueeze(0).unsqueeze(0))
            print(f"Encoded: {enc}")
            dec = autoenc.decoder(enc)
            
            ax[1,i].matshow(dec.squeeze().cpu().numpy())
            latent = enc.squeeze().cpu().numpy()
            ax[1,i].set_title("Latent representation: " + str(latent))
            
        # Load an image
        img = get_htc_scan(7, "a")
        img_distorted = img
        img_distorted = shuffle_local_pixels(img, area=patch_size // 2, shuffle_chance=0.4)
        img_distorted = random_distort(img_distorted, factor=0.1)
        
        img = torch.tensor(img, dtype=torch.float32)
        img_distorted = torch.tensor(img_distorted, dtype=torch.float32)
        print(f"Distorted image shape: {img_distorted.shape}")
        
        reconstructed = autoenc.remove_noise_from_img(img,
                                                      patch_size,
                                                      stride,
                                                      batch_size,
                                                      patches_to_device="cpu",
                                                      patches_to_dtype=torch.bool
                                                      )


        fig, ax = plt.subplots(1, 3)
        ax[0].matshow(img.cpu().numpy())
        ax[0].set_title("Original image")
        ax[1].matshow(img_distorted.cpu().numpy())
        ax[1].set_title("Distorted image")
        ax[2].matshow(reconstructed.cpu().numpy())
        ax[2].set_title("Reconstructed image")
    
    plt.show()
    
    


    
    
    
    