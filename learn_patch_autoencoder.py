import os
import random
import sys
import torch
import numpy as np
from torchvision.io import read_image, ImageReadMode
from PIL import Image
import argparse

# TESTING
from AbsorptionMatrices import Circle
import torch
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
    # nans to zeros
    image = torch.nan_to_num(image)
    image = image.to(device)
    return image

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.shortcut = (
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else torch.nn.Identity()
        )

    def forward(self, x):
        return torch.nn.ReLU()(self.block(x) + self.shortcut(x))

class PatchAutoencoder(torch.nn.Module):
    def __init__(self, patch_size = 16, num_latent = 8, pretrained_weights=None):
        super(PatchAutoencoder, self).__init__()
        self.encoder, self.decoder = self.get_autoencoder(patch_size, num_latent)
        if pretrained_weights:
            self.load_state_dict(torch.load(pretrained_weights))
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        return self
        
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
    
    def remove_noise_from_img(self, img, patch_size, stride, batch_size, patches_to_device='cuda', patches_to_dtype=torch.float32):
        """ Remove noise from an image using the autoencoder.
        """
        # Extract patches
        patches = extract_patches_2d_pt(img, patch_size, stride=stride, device=patches_to_device, dtype=patches_to_dtype)
        batch_size = batch_size if batch_size > 0 else len(patches)
        dec_patches = []
        # Encode in batches
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                #print(f"Batch shape: {batch.shape}")
                if patches_to_device != "cuda":
                    batch = batch.to("cuda")
                #if patches_to_dtype != torch.float32:
                #    batch = batch.to(torch.float32)
                dec = self(batch)
                dec_patches.append(dec.cpu())
        #print(f"Computed {len(dec_patches)} batches")
        dec_patches = torch.cat(dec_patches, dim=0)
        dec_patches = dec_patches.squeeze(1)
        # Reconstruct the image
        reconstructed = reconstruct_from_patches_2d_pt(dec_patches, img.shape, stride=stride)
        return reconstructed
    
class AdvancedPatchAutoencoder(PatchAutoencoder):
    
    def get_advanced_autoencoder(patch_size=16, num_latent=8):
        """
        Create an improved autoencoder model with a more advanced architecture for representing different types of patches.
        """

        encoder = torch.nn.Sequential(
            ResidualBlock(1, 32),  # Input: (1, patch_size, patch_size)
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample
            ResidualBlock(32, 64),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample
            ResidualBlock(64, 128),
            torch.nn.AdaptiveAvgPool2d(1),  # Aggregate spatial dimensions
            torch.nn.Flatten(),
            torch.nn.Linear(128, num_latent),  # Latent representation
        )

        decoder = torch.nn.Sequential(
            torch.nn.Linear(num_latent, 128),  # Decode latent representation
            torch.nn.Unflatten(1, (128, 1, 1)),  # Reshape to feature map
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResidualBlock(64, 64),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResidualBlock(32, 32),
            torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid(),  # Output pixel values between 0 and 1
        )
        return encoder, decoder

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
                  restore_best=True,
                  save_to="patch_autoencoder.pth",
                  overwrite=False):
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
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss}", end="\r")
        # Test the model
        with torch.no_grad():
            for i in range(0, len(test_filtered_patches), batch_size):
                batch = test_filtered_patches[i:i+batch_size]
                batch = torch.tensor(batch, dtype=torch.float32).to(torch.get_default_device())
                decoded = autoenc(batch)
                decoded = decoded.squeeze(1)
                test_loss = criterion(decoded, batch)
            print(f"Epoch {epoch}, Test loss: {test_loss}")
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
    if overwrite or not os.path.exists(save_to):
        torch.save(autoenc.state_dict(), save_to)
    elif not overwrite:
        print(f"Model file {save_to} already exists, not overwriting")
    return autoenc


if __name__ == "__main__":
    patch_size = 32
    num_epochs = 50
    patience = 5
    restore_best = True
    learning_rate = 0.001
    batch_size = 64
    train_test_split = 0.85
    load_pre_trained = ""#patch_autoencoder_P40_D10.pth"
    do_training = True
    overwrite = True
    test_stride = 1
    from_command_line = True if len(sys.argv) > 1 else False
    do_demo = True if not from_command_line else False
    
    parser = argparse.ArgumentParser(description='Patch Autoencoder')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of the patches')
    args = parser.parse_args()
    patch_size = args.patch_size
    num_latent = patch_size // 4
    train_stride = patch_size // 4
    
    save_to = f"patch_autoencoder_P{patch_size}_D{num_latent}.pth"
    
    torch.manual_seed(0)
    
    
    # Check that GPU is available
    if torch.cuda.is_available():
        print("Using GPU")
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')
    
    # Load images as numpy arrays, and use sklearn to learn the dictionary.
    #images = [get_basic_circle_image() for _ in range(10)]
    images = load_generated_images("generated_data8")    
    if images[0].device.type == 'cuda':
        images = [img.cpu().numpy() for img in images]
    if not do_training:
        images = random.choices(images, k = 3)
    print(f"Loading {len(images)} images")
    images = np.array(images)
    
    images = [torch.tensor(img, dtype=torch.float32) for img in images]
    image_patches = [extract_patches_2d_pt(img, patch_size,"cuda",dtype=torch.bool,stride=train_stride) for img in images]
    
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
                                restore_best=restore_best,
                                save_to=save_to,
                                overwrite=overwrite)
    
    if not do_demo:
        print(f"Training done, saved model to {save_to}")
        sys.exit(0)
    
    # Show 4 enc-dec examples
    fig, ax = plt.subplots(2, 4)
    # Make a big suptitle
    fig.suptitle(f"Patch autoencoder with patch size {patch_size} and {num_latent} latent variables",
                    fontsize=16)
    with torch.no_grad():
        for i in range(4):
            idx = np.random.randint(0, len(test_filtered_patches))
            original = test_filtered_patches[idx].clone().squeeze().cpu().numpy()
            original = original.astype(np.float32)
            while np.sum(original) < 0.1 * patch_size * patch_size or np.sum(original) > 0.8 * patch_size * patch_size:
                idx = np.random.randint(0, len(test_filtered_patches))
                original = test_filtered_patches[idx].clone().squeeze().cpu().numpy()
                original = original.astype(np.float32)
            original_distorted = original.copy()
            if i >= 2:
                # Distort the patch
                original_distorted = shuffle_local_pixels(original_distorted, area=patch_size//8, shuffle_chance=0.4)
                original_distorted = random_distort(original_distorted, factor=0.4)
            ax[0,i].matshow(original_distorted)
            ax[0,i].set_title("Input patch" if i < 2 else "Input patch\nwith artifacts")
            
            
            
            enc = autoenc.encode(torch.tensor(original_distorted).unsqueeze(0).unsqueeze(0))
            dec = autoenc.decoder(enc)
            
            ax[1,i].matshow(dec.squeeze().cpu().numpy())
            latent = enc.squeeze().cpu().numpy()
            mae = np.mean(np.abs(original - dec.squeeze().cpu().numpy()))
            ax[1,i].set_title(f"Encoded-Decoded\nMAE: {mae:.2f}")
            
            for a in ax[:,i]:
                a.set_xticks([])
                a.set_yticks([])
            
        # Load an image
        img = get_htc_scan(7, "a")
        img_distorted = img
        img_distorted = shuffle_local_pixels(img, area=patch_size // 2, shuffle_chance=0.4)
        img_distorted = random_distort(img_distorted, factor=0.2)
        
        img = torch.tensor(img, dtype=torch.float32)
        img_distorted = torch.tensor(img_distorted, dtype=torch.float32)
        print(f"Distorted image shape: {img_distorted.shape}")
        
        reconstructed = autoenc.remove_noise_from_img(img_distorted,
                                                      patch_size,
                                                      stride = test_stride,
                                                      batch_size=batch_size,
                                                      )
        
        img_cpu_np = img.cpu().numpy()
        img_distorted_cpu_np = img_distorted.cpu().numpy()
        reconstructed_cpu_np = reconstructed.cpu().numpy()
        # The reconstructed image might have nans, replace them with zeros
        reconstructed_cpu_np = np.nan_to_num(reconstructed_cpu_np)
        
        mae = np.mean(np.abs(reconstructed_cpu_np - img_cpu_np))
        print(f"MAE: {mae}")

        fig, ax = plt.subplots(1, 3)
        fig.suptitle(f"Removing artifacts using a {patch_size}x{patch_size}\npatch autoencoder with {num_latent} latent variables",
                    fontsize=16)
        ax[0].matshow(img_cpu_np)
        ax[0].set_title("Original image")
        ax[1].matshow(img_distorted_cpu_np)
        ax[1].set_title("Distorted image")
        ax[2].matshow(reconstructed_cpu_np)
        ax[2].set_title("Reconstructed image")
        # Remove ticks
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        # Show mae of reconstruction
        ax[2].set_title(f"Reconstructed image\nMAE: {mae:.2f}")
    
    
    plt.show()
    
    


    
    
    
    