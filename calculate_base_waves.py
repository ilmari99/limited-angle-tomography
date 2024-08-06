import numpy as np
import matplotlib.pyplot as plt
import torch
from AbsorptionMatrices import Circle
from utils import FBPRadon, filter_sinogram
from regularization import reconstruct_from_patches_2d_pt, extract_patches_2d_pt

def get_possible_pixels(image_size, patch_size=1, stride=1, skip_outside_circle=True):
    """ Return a list of pixels, where the pixel denotes where a patch of size 'patch_size' can be placed,
    s.t. the patch is fully inside the largest possible circle that can be placed inside the image.
    """
    possible_pixels = []
    inside_circle_mask = []
    for i in range(0, image_size, stride):
        for j in range(0, image_size, stride):
            patch_start = (i,j)
            patch_end = (i + patch_size, j + patch_size)
            # Both the start and end have to be inside the image
            if patch_end[0] >= image_size or patch_end[1] >= image_size:
                continue
            # Both the start and end have to be inside the circle
            # Check start
            start_dist = np.sqrt((patch_start[0] - image_size // 2) ** 2 + (patch_start[1] - image_size // 2) ** 2)
            end_dist = np.sqrt((patch_end[0] - image_size // 2) ** 2 + (patch_end[1] - image_size // 2) ** 2)
            if start_dist <= image_size // 2 and end_dist <= image_size // 2:
                possible_pixels.append((i, j))
                inside_circle_mask.append(1)
            # Now we know, that the patch is not fully inside the circle
            elif not skip_outside_circle:
                inside_circle_mask.append(0)
                possible_pixels.append((i, j))
    if not skip_outside_circle:
        return possible_pixels, inside_circle_mask
    
    return possible_pixels

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

def show_circle(image_size, circle_pixels, patch_size):
    """ Print a circle with radius 'size' and center at (size // 2, size // 2).
    """
    mat = np.zeros((image_size, image_size))
    for i, j in circle_pixels:
        mat[i:i+patch_size, j:j+patch_size] = 1
    fg, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(mat, cmap='gray')
    return fg, ax
    
def get_base_matrices(image_size, patch_size, stride=1, skip_outside_circle=True):
    """ Return the base matrices for the wave equation.
    """
    if skip_outside_circle:
        possible_pixels = get_possible_pixels(image_size, patch_size, stride=stride, skip_outside_circle=True)
    else:
        possible_pixels, inside_circle_mask = get_possible_pixels(image_size, patch_size, stride=stride, skip_outside_circle=False)
    mats = []
    for i, j in possible_pixels:
        mat = np.zeros((image_size, image_size))
        mat[i:i+patch_size, j:j+patch_size] = 1
        mats.append(mat)
    mats = np.array(mats)
    if not skip_outside_circle:
        return mats, inside_circle_mask
    return mats

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

def show_sinograms(images, sinograms, n=6):
    """ Plot the images, and their corresponding sinograms.
    """
    fig, axs = plt.subplots(2, n, figsize=(20, 10))
    chosen_idxs = np.random.choice(len(images), n)
    for fig_idx, sample_idx in enumerate(chosen_idxs):
        axs[0, fig_idx].imshow(images[sample_idx], cmap='gray')
        axs[1, fig_idx].imshow(sinograms[sample_idx], cmap='gray')
    fig.suptitle('So-called base matrices and their corresponding sinograms')
    return fig, axs

class FindSuperposition(torch.nn.Module):
    def __init__(self, base_matrices, sinograms, angles, a):
        super(FindSuperposition, self).__init__()
        self.base_matrices = torch.tensor(base_matrices, dtype=torch.float32, device='cuda')
        self.sinograms = torch.tensor(sinograms, dtype=torch.float32, device='cuda')
        self.angles = angles
        self.a = a
        self.weights = torch.nn.Parameter(torch.rand(len(base_matrices), requires_grad=True))
        
    def forward(self):
        # The reconstructed sinogram is the sum of the weighted base sinograms
        reconstructed_sinogram = torch.zeros_like(self.sinograms[0])
        for i, weight in enumerate(self.weights):
            reconstructed_sinogram += weight * self.sinograms[i]
        return reconstructed_sinogram
    
class SinogramToPatchesConnected(torch.nn.Module):
    def __init__(self, image_size, patch_size, stride, angles, a=0):
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
        #print(f"Image size: {image_size}, Patch size: {patch_size}, Stride: {stride}")
        
        self.radont = FBPRadon(image_size, angles, a=a)
        
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
        self.sinogram_masks = torch.stack(masks).to("cpu").to(torch.uint8)
        print(f"Masks: {self.sinogram_masks.shape}")
        
        # We use a single model to predict each patch, based on it's masked sinogram
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(len(self.angles) * self.image_size, self.patch_size * self.patch_size),
            #torch.nn.ReLU(),
            #torch.nn.Linear(2 *self.patch_size * self.patch_size, self.patch_size * self.patch_size),
            #torch.nn.Sigmoid()
        )
        self.model.to('cuda')
    
    def forward(self, sinogram):
        # Get the masked sinograms
        sinogram = sinogram.to('cpu')
        # Elementwise multiplication
        masked_sinograms = sinogram * self.sinogram_masks
        #print(f"Masked sinograms shape: {masked_sinograms.shape}")
        # As input, give a masked sinogram, and predict a patch_size x patch_size patch
        # npatches x 1 x num_angles x image_size
        masked_sinograms = masked_sinograms.unsqueeze(1)
        patches = []
        # Do in batchs of 64
        batch_size = 32
        for i in range(0, masked_sinograms.shape[0], batch_size):
            n = batch_size if i + batch_size < masked_sinograms.shape[0] else masked_sinograms.shape[0] - i
            batch = masked_sinograms[i:i+n]
            batch = batch.to('cuda').float()
            patch = self.model(batch)
            patch = torch.reshape(patch, (n, self.patch_size, self.patch_size))
            patches.append(patch.cpu().to(torch.float16))
        patches = torch.cat(patches)
        patches = patches.squeeze()
        #print(f"Patches shape: {patches.shape}")
        #patches = patches.squeeze()
        # Reconstruction from patches
        y_hat = reconstruct_from_patches_2d_pt(patches, (self.image_size, self.image_size), stride=self.stride, device='cpu')
        y_hat = y_hat.to('cuda').float()
        y_hat = torch.sigmoid(y_hat)
        s_hat = self.radont.forward(y_hat)
        return y_hat, s_hat
    
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


###############################################################################################
###############################################################################################
###############################################################################################

##### TESTING #####
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

def get_basic_circle_scan(angles):
    if angles is None:
        angles = np.linspace(0,60,60, endpoint=False)
    circle = create_circle()
    y = circle.matrix
    y = torch.tensor(y, dtype=torch.float32, device='cuda')
    y = torch.nn.functional.pad(y, (0,1,0,1))
    y = torch.clip(y, 0, 1)
    y = torch.round(y)
    return y

###############################################################################################
###############################################################################################
###############################################################################################


torch.autograd.set_detect_anomaly(True)
angles = np.deg2rad(np.arange(0, 60, 1))
a = 5.5
patch_size = 5
image_size = 128
stride = 5
rt = FBPRadon(image_size, angles, a)

example_circle = get_basic_circle_scan(angles)
example_sinogram = rt.forward(example_circle)

model = SinogramToPatchesConnected(image_size, patch_size, stride, angles, a)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.L1Loss()
n_epochs = 100
losses = []
for epoch in range(n_epochs):
    optimizer.zero_grad()
    y_hat, s_hat = model(example_sinogram)
    loss = criterion(example_sinogram, s_hat)

    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch}, loss: {loss.item()}")


y_hat, s_hat = model(example_sinogram)
reconstructed_circle = y_hat.detach().cpu().numpy()
reconstructed_sinogram = s_hat.detach().cpu().numpy()


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(example_circle.cpu().numpy(), cmap='gray')
ax[0].set_title('Original circle')
ax[1].imshow(reconstructed_circle, cmap='gray')
ax[1].set_title('Reconstructed circle')


loss_fig, loss_ax = plt.subplots(1, 1, figsize=(10, 5))
loss_ax.plot(losses)
loss_ax.set_xlabel('Epoch')
loss_ax.set_ylabel('Loss')
loss_ax.set_title('Loss during optimization')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
example_sinogram = example_sinogram.cpu().numpy()
axs[0].imshow(example_sinogram, cmap='gray')
axs[0].set_title('Original sinogram')
axs[1].imshow(reconstructed_sinogram, cmap='gray')
axs[1].set_title('Reconstructed sinogram')

plt.show()




