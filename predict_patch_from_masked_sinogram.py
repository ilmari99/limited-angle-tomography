import torch
import numpy as np
from AbsorptionMatrices import Circle
from utils import FBPRadon
import math
from utils import extract_patches_2d_pt, reconstruct_from_patches_2d_pt
from pytorch_models import EncoderDecoder
import sympy
import matplotlib.pyplot as plt

class Decoder(torch.nn.Module):
    def __init__(self, encoder_output_elems, output_shape_side_len, filters = [1, 64, 32,1]):
        super(Decoder, self).__init__()
        self.encoder_output_elems = encoder_output_elems
        self.encoder_output_side_len = math.sqrt(encoder_output_elems)
        self.filters = filters
        assert self.encoder_output_side_len.is_integer()
        self.encoder_output_side_len = int(self.encoder_output_side_len)
        self.output_shape_side_len = output_shape_side_len
        self.num_conv_layers = len(filters) - 1
        parameters = self.find_parameters()
        self.kernel_sizes, self.paddings, self.strides = parameters
        self.decoder = self.create_decoder()
        
    def create_decoder(self):
        layers = []
        filters = self.filters
        for i in range(self.num_conv_layers):
            if i == 0:
                layers.append(torch.nn.ConvTranspose2d(filters[i],
                                                       filters[i+1],
                                                       kernel_size=self.kernel_sizes[i],
                                                       stride=self.strides[i],
                                                       padding=self.paddings[i]))
            else:
                layers.append(torch.nn.ConvTranspose2d(filters[i],
                                                       filters[i+1],
                                                       kernel_size=self.kernel_sizes[i],
                                                       stride=self.strides[i],
                                                       padding=self.paddings[i]))
            layers.append(torch.nn.ReLU())
        decoder = torch.nn.Sequential(*layers)
        return decoder
        
        
    def find_parameters(self):
        # We need to find a solution for parameters, such that the
        # output is (output_shape_side_len, output_shape_side_len)
        # The dimension after a convolutional layer can be calculated as
        # d_i+1 = [(d_i−K_i+2P_i)/S_i]+1
        # We need a parameter for each kernel size(w,h), padding(w,h) and stride(w,h)
                
        # We can define a system of equations, where we solve for the parameters
        # based on
        # d_1h = [(d_0h−K_0h+2P_0h)/S_0h]+1
        # d_1w = [(d_0w−K_0w+2P_0w)/S_0w]+1
        # d_2h = [(d_1h−K_1h+2P_1h)/S_1h]+1
        # ...
        # d_nh = [(d_{n-1}h−K_{n-1}h+2P_{n-1}h)/S_{n-1}h]+1
        # where d_0h, d_0w = encoder_output_side_len
        # d_nh = output_shape_side_len
        
        system_of_equations = []
        for i in range(self.num_conv_layers):
            #system_of_equations.append(sympy.Eq(sympy.symbols(f"d_{i+1}h"),
            #                                    ((sympy.symbols(f"d_{i}h") - sympy.symbols(f"K_{i}h") + 2 * sympy.symbols(f"P_{i}h")) / sympy.symbols(f"S_{i}h")) + 1))
            # For transposed convolutions:
            eq = ((sympy.symbols(f"d_{i}h") - 1) * sympy.symbols(f"S_{i}h") - 2 * sympy.symbols(f"P_{i}h") + sympy.symbols(f"K_{i}h"))
            system_of_equations.append(sympy.Eq(sympy.symbols(f"d_{i+1}h"), eq))
            #system_of_equations.append(sympy.Eq(sympy.symbols(f"d_{i+1}w"),
            #                                    ((sympy.symbols(f"d_{i}w") - sympy.symbols(f"K_{i}w") + 2 * sympy.symbols(f"P_{i}w")) / sympy.symbols(f"S_{i}w")) + 1))
        # Add the constraints
        system_of_equations.append(sympy.Eq(sympy.symbols(f"d_0h"), self.encoder_output_side_len))
        #system_of_equations.append(sympy.Eq(sympy.symbols(f"d_0w"), self.encoder_output_side_len))
        system_of_equations.append(sympy.Eq(sympy.symbols(f"d_{self.num_conv_layers}h"), self.output_shape_side_len))
        #system_of_equations.append(sympy.Eq(sympy.symbols(f"d_{self.num_conv_layers}w"), self.output_shape_side_len))
        
        # Find an integer solution with Diophatine
        solution = sympy.solve(system_of_equations)
        print(solution)
        solution = solution[0]
        all_keys = []
        for layer_num in range(self.num_conv_layers):
            for wh in ['h']:
                all_keys.append(f"K_{layer_num}{wh}")
                all_keys.append(f"P_{layer_num}{wh}")
                all_keys.append(f"S_{layer_num}{wh}")
                all_keys.append(f"d_{layer_num+1}{wh}")
                
        all_keys = [sympy.symbols(key) for key in all_keys]
        free_variables = set(all_keys) - set(solution.keys())
        print(f"Free variables: {free_variables}")
        # The solution has a mapping from sym -> eq,
        # where the free variables define the solution
        # Let's find the solution where all the free variables are 1
        subs_map = {"K" : 3, "P" : 0, "S" : 1, "d" : self.output_shape_side_len}
        # Add the free variables to the solution
        for free_var in free_variables:
            value = subs_map[free_var.name[0]]
            solution[free_var] = value
        # Now solve the system of equations for the remaining variables
        system_of_equations = [sympy.Eq(sym, eq) for sym, eq in solution.items()]
        solution = sympy.solve(system_of_equations)
        print(f"Solution: {solution}")
        solution = solution[0]
        solution = {str(key): value for key, value in solution.items()}
        
        kernel_sizes = [(solution[f"K_{i}h"], solution[f"K_{i}h"]) for i in range(self.num_conv_layers)]
        paddings = [(solution[f"P_{i}h"], solution[f"P_{i}h"]) for i in range(self.num_conv_layers)]
        strides = [(solution[f"S_{i}h"], solution[f"S_{i}h"]) for i in range(self.num_conv_layers)]
        print(f"Kernel sizes: {kernel_sizes}, Paddings: {paddings}, Strides: {strides}")
        if self.check_parameters(kernel_sizes, paddings, strides):
            print(f"Check passed")
        else:
            print(f"Check failed")
        return kernel_sizes, paddings, strides
    
        
        
    def check_parameters(self, kernel_sizes, paddings, strides):
        """ Check that the output is (output_shape_side_len, output_shape_side_len)
        """
        assert len(kernel_sizes) == self.num_conv_layers
        assert len(paddings) == self.num_conv_layers
        assert len(strides) == self.num_conv_layers
        # Check that the output is a square
        # Tuples of (input_height, output_height) and (input_width, output_width)
        height_dim_input_and_outputs = []
        width_dim_input_and_outputs = []
        for layer_idx in range(self.num_conv_layers):
            # For the first layer, the input is the encoder output
            height_transformation = []
            width_transformation = []
            if layer_idx == 0:
                height_transformation.append(self.encoder_output_side_len)
                width_transformation.append(self.encoder_output_side_len)
            else:
                height_transformation.append(height_dim_input_and_outputs[-1][1])
                width_transformation.append(width_dim_input_and_outputs[-1][1])
            input_height = height_transformation[0]
            input_width = width_transformation[0]
            # Calculate the output height and width
            new_height = ((input_height - kernel_sizes[layer_idx][0] + 2 * paddings[layer_idx][0]) / strides[layer_idx][0]) + 1
            new_width = ((input_width - kernel_sizes[layer_idx][1] + 2 * paddings[layer_idx][1]) / strides[layer_idx][1]) + 1
            
            height_transformation.append(new_height)
            width_transformation.append(new_width)
            height_dim_input_and_outputs.append(height_transformation)
            width_dim_input_and_outputs.append(width_transformation)
        # Check that the final width and height are output_shape_side_len
        if height_dim_input_and_outputs[-1][1] != self.output_shape_side_len:
            return False
        if width_dim_input_and_outputs[-1][1] != self.output_shape_side_len:
            return False
        return True

    def forward(self, x):
        # Firstly, we reshape the input to a square
        x = x.reshape((1, 1, self.encoder_output_side_len, self.encoder_output_side_len))
        # Then we pass it through the decoder
        x = self.decoder(x)
        # reshape it to the output shape
        x = x.reshape(-1, self.output_shape_side_len, self.output_shape_side_len)
        return x

class PredictPatchFromMaskedSinogram(torch.nn.Module):
    def __init__(self, input_shape, angles, patch_size):
        """ A model that takes in a sinogram that is masked, so only the relevant region is visible.
        """
        super(PredictPatchFromMaskedSinogram, self).__init__()
    
        self.input_shape = input_shape
        self.angles = angles
        self.patch_size = patch_size
        self.rt = FBPRadon(input_shape[1], angles, a=0)
        
        # Encoder-decoder architecture
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(input_shape[1] * input_shape[2], 36),
        )
        
        self.decoder = Decoder(self.get_encoder_output_shape(input_shape)[0], patch_size)
        
    def get_encoder_output_shape(self, input_shape):
        with torch.no_grad():
            x = torch.randn(input_shape)
            x = self.encoder(x)
        x = x.squeeze()
        # The output size must be a single square number
        assert math.sqrt(x.shape[0]).is_integer()
        return x.shape
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x
        

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
    # Clip the values to [0, 1]
    y = np.clip(y, 0, 1)
    return y



image_size = 128
patch_size = 10
patch_stride = 10
angles = np.deg2rad(np.linspace(0, 180, 180))
rt = FBPRadon(image_size, angles, a=0)
rt_a_zero = FBPRadon(image_size, angles, a=0)
torch.set_default_device('cuda')

patch_start_pixels = get_patch_starts(image_size, patch_size, patch_stride)
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
    
# Find the sinogram of each mask
patch_sinograms = []
for mask in patch_masks:
    mask = torch.tensor(mask, dtype=torch.float32, device='cuda')
    sinogram = rt_a_zero.forward(mask)
    sinogram = sinogram.cpu()
    patch_sinograms.append(sinogram)

# Find a mask for the singoram, for each img x img patch,
# where areas of the singoram relevant to the patch are 1
patch_sinogram_masks = []
for sinogram in patch_sinograms:
    mask = torch.zeros_like(sinogram)
    mask[torch.abs(sinogram) > 1e-6] = 1
    patch_sinogram_masks.append(mask)
patch_sinogram_masks = torch.stack(patch_sinogram_masks).to('cuda')

train_circles = [get_basic_circle_image() for _ in range(500)]
test_circles = [get_basic_circle_image() for _ in range(10)]
train_circles_pt = [torch.tensor(circle, dtype=torch.float32) for circle in train_circles]
test_circles_pt = [torch.tensor(circle, dtype=torch.float32) for circle in test_circles]

train_circle_patches_pt = [extract_patches_2d_pt(circle, patch_size, stride=patch_stride) for circle in train_circles_pt]
test_circle_patches_pt = [extract_patches_2d_pt(circle, patch_size, stride=patch_stride) for circle in test_circles_pt]
train_circle_sinograms = [rt.forward(circle) for circle in train_circles_pt]
test_circle_sinograms = [rt.forward(circle) for circle in test_circles_pt]
print(f"Number of patches in a circle according to extract_patches_2d: {len(train_circle_patches_pt[0])}")
print(f"Number of patches in a circle according to the formula: {len(patch_start_pixels)}")

get_patch = 20
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(train_circle_patches_pt[0][get_patch].cpu().numpy())
ax[0].set_title("Patch")
ax[1].imshow(patch_masks[get_patch] * train_circles_pt[0].cpu().numpy())
ax[1].set_title("Mask")
# Create a dataset generator
# The generator selects a random circle, and a random patch from the circle
# It then finds the masked sinogram for that patch, and
# yields the masked sinogram and the patch of the circle

# Plot a circle
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(train_circles_pt[0].cpu().numpy())
ax.set_title("Circle")

def dataset_generator(circles, circle_patches_pt, circle_sinograms):
    global patch_sinogram_masks
    while True:
        circle_idx = np.random.randint(0, len(circles))
        patch_idx = np.random.randint(0, len(circle_patches_pt[0]))
        #print(f"Circle idx: {circle_idx}, Patch idx: {patch_idx}")
        masked_sinogram = circle_sinograms[circle_idx] * patch_sinogram_masks[patch_idx]
        masked_sinogram = masked_sinogram.unsqueeze(0)
        patch = circle_patches_pt[circle_idx][patch_idx].unsqueeze(0)
        yield masked_sinogram, patch

# Create the generator
train_gen = dataset_generator(train_circles_pt, train_circle_patches_pt, train_circle_sinograms)
test_gen = dataset_generator(test_circles_pt, test_circle_patches_pt, test_circle_sinograms)

# Plot some samples from the dataset
for _ in range(4):
    masked_sinogram, patch = next(train_gen)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(masked_sinogram.cpu().squeeze().numpy())
    ax[0].set_title("Masked Sinogram")
    ax[1].imshow(patch.cpu().squeeze().numpy())
    ax[1].set_title("Patch")
#plt.show()
#exit()

model = PredictPatchFromMaskedSinogram((1, len(angles), image_size), angles, patch_size)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Train the model
epochs = 10
train_losses = []
test_losses = []
best_loss = float('inf')
patience = 6
counter = 0
batch_size = 64
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    num_batches = 1000 // batch_size
    for _ in range(num_batches):
        optimizer.zero_grad()
        batch_loss = 0.0
        for _ in range(batch_size):
            masked_sinogram, patch = next(train_gen)
            masked_sinogram = masked_sinogram.cuda()
            patch = patch.cuda()
            output = model(masked_sinogram)
            loss = loss_fn(output, patch)
            loss.backward()
            batch_loss += loss.item()
        optimizer.step()
        train_loss += batch_loss / batch_size
    train_loss /= num_batches
    train_losses.append(train_loss)
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i in range(10):
            masked_sinogram, patch = next(test_gen)
            masked_sinogram = masked_sinogram.cuda()
            patch = patch.cuda()
            output = model(masked_sinogram)
            loss = loss_fn(output, patch)
            test_loss += loss.item()
    test_loss /= 10
    test_losses.append(test_loss)
    
    print(f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")
    
    if test_loss < best_loss:
        best_loss = test_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break

for _ in range(4):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    masked_sinogram, patch = next(test_gen)
    masked_sinogram = masked_sinogram.cuda()
    patch = patch.cuda()
    output = model(masked_sinogram)
    output = output.cpu().detach().numpy()
    output = output.squeeze()
    ax[0].imshow(masked_sinogram.cpu().squeeze().numpy())
    ax[0].set_title("Masked Sinogram")
    ax[1].imshow(patch.cpu().squeeze().numpy())
    ax[1].set_title("Original Patch")
    ax[2].imshow(output)
    ax[2].set_title("Predicted Patch")
plt.show()

