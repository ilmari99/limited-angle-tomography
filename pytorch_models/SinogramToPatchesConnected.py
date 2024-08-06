import numpy as np
import torch
from utils import FBPRadon, reconstruct_from_patches_2d_pt


class SinogramToPatchesConnected(torch.nn.Module):
    def __init__(self, image_size, patch_size, stride, angles, batch_size=32):
        """ A model that uses a connected architecture to predict
        an image from a sinogram.
        The model is only locally connected, so that only the relevant
        part of the sinogram is used to predict the patch at a certain location.
        Args:   
            image_size: The size of the image
            patch_size: The size of a patch, i.e. the local receptive field
            stride: The stride of the patches. Overlapping patches are averaged.
            angles: The angles (rad) used in the Radon transform
        """
        super(SinogramToPatchesConnected, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.angles = angles
        self.batch_size = batch_size
        
        # Get the base_matrices and the corresponding sinograms
        patch_start_pixels = self.get_patch_starts(image_size, patch_size, stride=stride)
        #print(f"Number of patch_starts: {len(patch_start_pixels)}")
        
        #num_patches = extract_patches_2d_pt(torch.zeros((image_size, image_size), device='cuda'), patch_size, stride=stride)
        #print(f"Number of patches: {num_patches.shape}", flush=True)
        
        # Calculate which patches are actually inside the circle List[bool]
        patch_is_inside_circle = []
        for start in patch_start_pixels:
            i, j = start
            dist_from_center_to_start = np.sqrt((i - image_size // 2) ** 2 + (j - image_size // 2) ** 2)
            dist_from_center_to_end = np.sqrt((i + patch_size - image_size // 2) ** 2 + (j + patch_size - image_size // 2) ** 2)
            if dist_from_center_to_start <= image_size // 2 and dist_from_center_to_end <= image_size // 2:
                patch_is_inside_circle.append(True)
            else:
                patch_is_inside_circle.append(False)
        print(f"Num inside circle: {sum(patch_is_inside_circle)}, Num outside circle: {len(patch_is_inside_circle) - sum(patch_is_inside_circle)}")
        # Find every img x img mask, where only the patch is 1
        patch_masks = []
        for patch_idx, start in enumerate(patch_start_pixels):
            i, j = start
            mask = np.zeros((image_size, image_size))
            if patch_is_inside_circle[patch_idx]:
                mask[i:i+patch_size, j:j+patch_size] = 1
            patch_masks.append(mask)
        
        patch_masks = np.array(patch_masks)
        base_sinograms = self.get_base_sinograms(patch_masks, angles)
        #self.base_sinograms.to("cpu")
        #self.patch_masks = torch.tensor(self.patch_masks, dtype=torch.float32, device='cpu')
        # TODO: The values at base_sinograms could actually be used as sort of attention weights
        
        # The sinogram masks are the sinograms, but every != 0 value is set to 1
        masks = []
        for sinogram in base_sinograms:
            mask = torch.where(sinogram > 1e-6, 1, 0)
            masks.append(mask)
        avg_num_of_ones_in_mask = sum([torch.sum(mask).item() for mask in masks]) / len(masks)
        print(f"Avg num of ones in mask: {avg_num_of_ones_in_mask}")
        self.sinogram_masks = torch.stack(masks).to("cpu").to(torch.float16)
        print(f"Masks: {self.sinogram_masks.shape}")
        
        # We use a single model to predict each patch, based on it's masked sinogram
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(len(self.angles) * self.image_size, self.patch_size * self.patch_size),
        )
        self.model.to('cuda')
    
    def forward(self, sinogram):
        # Get the masked sinograms
        sinogram = sinogram.to('cpu').to(torch.float16)
        # Elementwise multiplication
        masked_sinograms = sinogram * self.sinogram_masks
        #print(f"Masked sinograms shape: {masked_sinograms.shape}")
        # As input, give a masked sinogram, and predict a patch_size x patch_size patch
        # npatches x 1 x num_angles x image_size
        masked_sinograms = masked_sinograms.unsqueeze(1)
        patches = []
        for i in range(0, masked_sinograms.shape[0], self.batch_size):
            n = self.batch_size if i + self.batch_size < masked_sinograms.shape[0] else masked_sinograms.shape[0] - i
            batch = masked_sinograms[i:i+n]
            batch = batch.to('cuda').float()
            patch_batch = self.model(batch)
            patch_batch = torch.reshape(patch_batch, (n, self.patch_size, self.patch_size))
            patches.append(patch_batch.cpu().to(torch.float16))
        patches = torch.cat(patches)
        patches = patches.squeeze()
        #print(f"Patches shape: {patches.shape}")
        #patches = patches.squeeze()
        # Reconstruction from patches
        y_hat = reconstruct_from_patches_2d_pt(patches, (self.image_size, self.image_size), stride=self.stride, device='cpu')
        y_hat = y_hat.to('cuda').float()
        y_hat = torch.sigmoid(y_hat)
        return y_hat
    
    @staticmethod
    def get_patch_starts(image_size, patch_size, stride=1):
        """ Return the starting pixel of each patch.
        """
        patch_starts = []
        for i in range(0, image_size, stride):
            for j in range(0, image_size, stride):
                if i + patch_size > image_size or j + patch_size > image_size:
                    continue
                patch_starts.append((i, j))
        return patch_starts
    
    @staticmethod
    def get_base_sinograms(base_matrices, angles, a=0) -> list:
        """ Get the sinograms of the base matrices.
        """
        rt = FBPRadon(base_matrices.shape[1], angles, a)
        base_sinograms = []
        for mat in base_matrices:
            mat = torch.tensor(mat, dtype=torch.float32, device='cuda')
            sinogram = rt.forward(mat)
            sinogram = sinogram.cpu()
            base_sinograms.append(sinogram)
        return base_sinograms