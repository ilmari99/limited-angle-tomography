import warnings
import numpy as np
import cv2
import torch
from torch_radon import Radon, RadonFanbeam

def filter_sinogram(sino, a=0.1, device='cuda'):
    """filter projections. Normally a ramp filter multiplied by a window function is used in filtered
    backprojection. The filter function here can be adjusted by a single parameter 'a' to either approximate
    a pure ramp filter (a ~ 0)  or one that is multiplied by a sinc window with increasing cutoff frequency (a ~ 1).
    Credit goes to Wakas Aqram. 
    inputs: sino - [n x m] torch tensor where n is the number of projections and m is the number of angles used.
    outputs: filtSino - [n x m] filtered sinogram tensor
    
    Reference: https://github.com/csheaff/filt-back-proj
    """
    sino = torch.squeeze(sino)
    sino = sino.T
    
    projLen, numAngles = sino.shape
    step = 2 * np.pi / projLen
    w = torch.arange(-np.pi, np.pi, step, device=device)
    if len(w) < projLen:
        w = torch.cat([w, w[-1] + step])
    
    rn1 = abs(2 / a * torch.sin(a * w / 2))
    rn2 = torch.sin(a * w / 2) / (a * w / 2)
    r = rn1 * (rn2) ** 2
    filt = torch.fft.fftshift(r)
    # TODO: Check if this is correct
    filt[0] = 0
    filtSino = torch.zeros((projLen, numAngles), device=device, requires_grad=False)
    
    for i in range(numAngles):
        projfft = torch.fft.fft(sino[:, i])
        filtProj = projfft * filt
        ifft_filtProj = torch.fft.ifft(filtProj)
        
        filtSino[:, i] = torch.real(ifft_filtProj)

    return filtSino.T

class FBPRadonFanbeam(RadonFanbeam):
    def __init__(self, resolution, angles, a = 0.1, device='cuda', **kwargs):
        self.device = device
        super(FBPRadonFanbeam, self).__init__(resolution=resolution, angles=angles, source_distance=410.66, det_distance=553.74, **kwargs)
        #self.a = a
        self.a = torch.tensor(a, device=device, dtype=torch.float32, requires_grad=False)
        
    def parameters(self):
        return [self.a]
        
    def forward(self, x):
        s = super().forward(x)
        #self.a = torch.abs(self.a)
        if torch.equal(self.a, torch.tensor(0.0, device=self.device)):
            return s
        s = filter_sinogram(s, a = self.a, device=self.device)
        return s
        
class FBPRadon(Radon):
    def __init__(self, resolution, angles, a = 0.1, device='cuda',scale_sinogram=False, **kwargs):
        self.device = device
        super(FBPRadon, self).__init__(resolution=resolution, angles=angles, **kwargs)
        #self.filter = FourierFilters.construct_fourier_filter(size=resolution, filter_name='ramp')
        #print(f"Filter shape: {self.filter.shape}")
        self.a = a
        self.scale_sinogram = scale_sinogram
        self.a = torch.tensor(a, device=device, dtype=torch.float32, requires_grad=False)
        
    def parameters(self):
        return [self.a]
        
    def forward(self, x):
        s = super().forward(x)
        #self.a = torch.abs(self.a)
        if torch.equal(self.a, torch.tensor(0.0, device=self.device)):
            if self.scale_sinogram:
                s = (s - torch.min(s)) / (torch.max(s) - torch.min(s))
            return s
        s = filter_sinogram(s, a = self.a, device=self.device)
        if self.scale_sinogram:
            s = (s - torch.mean(s)) / torch.std(s)
        return s
    
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
            dec = self(batch).detach()
            dec = dec.to(patches_to_device)
            dec = dec.to(patches_to_dtype)
            dec_patches.append(dec)
        dec_patches = torch.cat(dec_patches, dim=0)
        dec_patches = dec_patches.squeeze(1)
        # Reconstruct the image
        reconstructed = reconstruct_from_patches_2d_pt(dec_patches, img.shape, stride=stride, device=patches_to_device)
        return reconstructed
    
def extract_patches_2d_pt(image, patch_size=8, device=None, dtype=None, stride=1):
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
    
    #print(f"Patches shape: {patches.shape}", flush=True)
    patches = patches.reshape(-1, patch_size*patch_size)
    patches = patches.unsqueeze(0).permute(0, 2, 1)
    #print(f"Patches shape: {patches.shape}", flush=True)
    counter_image = torch.ones_like(patches, device=device)
    #counter_image = torch.nn.functional.unfold(counter_image.unsqueeze(0).unsqueeze(0), patch_size, stride=1)
    #print(f"Counter image shape: {counter_image.shape}", flush=True)
    # Fold patches of ones to know how many patches overlap each pixel
    counter_image = torch.nn.functional.fold(counter_image, image_size, patch_size, stride=stride)
    counter_image = counter_image.squeeze()
    # HOW ON EARTH CAN THE COUNTER IMAGE BE 0??
    if torch.any(counter_image == torch.tensor(0, device=device)):
        warnings.warn("Counter image is 0")
        counter_image = counter_image + 1e-6
    #print(counter_image, flush=True)
    #print(f"Counter image shape: {counter_image.shape}")
    # fold the patches
    image = torch.nn.functional.fold(patches, image_size, patch_size, stride=stride)
    image = image.squeeze()
    image = image.to(device)
    # Divide the image by the counter image
    image = image / counter_image
    return image

def remove_noise_from_measurements(measurements):
    # Remove noise from the image
    # Make values with an absolute value less than 0.1 equal to 0
    m_copy = measurements.copy()
    m_copy_uint8 = m_copy.astype(np.uint8)
    #m_copy[np.abs(m_copy) < 0.1] = 0
    # Smooth the measurements and disconnect most of the noise from the main component
    kernel = np.ones((3,3),np.uint8)
    m_copy = cv2.morphologyEx(m_copy_uint8, cv2.MORPH_OPEN, kernel)
    
    # Only keep components that are connected to the largest component
    # Find the connected components
    _, labels = cv2.connectedComponents(m_copy_uint8)
    # Find the unique labels
    unique_labels = np.unique(labels)
    # Find the area of each label
    areas = [np.sum(labels == label) for label in unique_labels]
    # All of the waves will cross, so we only want to keep the largest area
    largest_area = np.max(areas)
    # Calculate a mask that is 1 where the largest area is, and 0 otherwise
    m_copy = np.where(labels == unique_labels[np.argmax(areas)], m_copy, 0)
    print(f"M_copy shape: {m_copy.shape}")
    
    # Take the original measurements only, where m_copy is 1
    measurements = measurements * m_copy
    
    return measurements

def reconstruct_error(m, m_hat):
    """ Calculate the l1 norm
    """
    return np.linalg.norm(m - m_hat, ord=1)