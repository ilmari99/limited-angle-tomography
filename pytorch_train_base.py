import numpy as np
import torch
from collections import OrderedDict

import torch.nn as nn
from torch_radon import Radon
import matplotlib.pyplot as plt

from AbsorptionMatrices import AbsorptionMatrix, Circle
from reconstruct import backproject_with_distance_measures, reconstruct_outer_shape, backproject_to_shape
import os
import torch
from torch.utils.data import Dataset

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, image_size=128):
        """ UNet model for image reconstruction
        
        Args:
            in_channels: int
                Number of input channels
            out_channels: int
                Number of output channels
            init_features: int
                Number of features in the first layer
            image_size: int
        """
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 2, features * 4, name="bottleneck")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        #x = x.unsqueeze(1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.upconv2(bottleneck)
        dec1 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.decoder2(dec1)
        dec0 = self.upconv1(dec1)
        dec0 = torch.cat((dec0, enc1), dim=1)
        dec0 = self.decoder1(dec0)
        return torch.sigmoid(self.conv(dec0))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

def get_outer_mask(shape : AbsorptionMatrix, angles, zero_threshold=0.1) -> torch.Tensor:
    """ Get the outer mask of the shape
    The mask describes the shape of the object based on laser distance measurements
    """
    measurements,distances_from_front,distances_from_back = shape.get_multiple_measurements(angles, return_distances=True, zero_threshold=0.1)
    thicknesses = np.full(measurements.shape, measurements.shape[1])
    thicknesses = thicknesses - distances_from_front - distances_from_back
    outer_mask = reconstruct_outer_shape(angles,distances_from_front,distances_from_back,zero_threshold=0.1)
    outer_mask = torch.tensor(outer_mask, dtype=torch.float32, device='cuda')
    # Pad to 128x128
    outer_mask = torch.nn.functional.pad(outer_mask, (0,1,0,1))
    return outer_mask

def sample_handler(measurements_path, shape_path, num_angles_limit=(30,120)):
    """ Given a path to the sinogram file, and the true shape file,
    this function:
    - Loads the sinogram,
    - Loads the true shape,
    - Picks random angles (total number between num_angles_limit),
    - Reconstructs the shape from the sinogram using the random angles,
    - Returns the reconstructed shape and the true shape as PyTorch tensors.
    """
    measurements = np.load(measurements_path)
    shape = np.load(shape_path)
    
    # Randomly choose angles
    num_angles = np.random.randint(num_angles_limit[0], num_angles_limit[1])
    angles = np.random.choice(np.arange(0,180), num_angles, replace=False)
    
    distances_from_front = measurements[angles,:,1].astype(np.int32)
    distances_from_back = measurements[angles,:,2].astype(np.int32)
    measurements = measurements[angles,:,0]
    
    thicknesses = np.full(measurements.shape, measurements.shape[1])
    thicknesses = thicknesses - distances_from_front - distances_from_back
    #measurements = thicknesses - measurements
    
    reconstr_shape = backproject_with_distance_measures(measurements,
                                                            angles,
                                                            distances_from_front,
                                                            distances_from_back,
                                                            use_filter=False,
                                                            zero_threshold=0.1
                                                            )
    reconstruction = reconstr_shape.matrix
    reconstruction = torch.tensor(reconstruction, dtype=torch.float32, device='cuda')
    # Return the reconstructed shape and the true shape
    shape = torch.tensor(shape, dtype=torch.float32, device='cuda')
    reconstruction = reconstruction.unsqueeze(0)
    shape = shape.unsqueeze(0)
    return reconstruction, shape

def filter_sinogram(sino, a = 0.1):
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
    #print(f"Sinogram shape: {sino.shape}")
    
    projLen, numAngles = sino.shape
    step = 2*np.pi/projLen
    w = torch.arange(-np.pi, np.pi, step, device='cuda')
    if len(w) < projLen:
        w = torch.cat([w, w[-1]+step]) #depending on image size, it might be that len(w) =  
                                        #projLen - 1. Another element is added to w in this case
    #print(w)
    rn1 = abs(2/a*torch.sin(a*w/2))  #approximation of ramp filter abs(w) with a funciton abs(sin(w))
    rn2 = torch.sin(a*w/2)/(a*w/2)   #sinc window with 'a' modifying the cutoff freqs
    r = rn1*(rn2)**2                 #modulation of ramp filter with sinc window
    filt = torch.fft.fftshift(r)
    # The first element in filt is Nan
    filt[0] = 0
    filtSino = torch.zeros((projLen, numAngles), device='cuda')
    
    for i in range(numAngles):
        projfft = torch.fft.fft(sino[:,i])
        filtProj = projfft*filt
        #print(f"Filt proj shape: {filtProj.shape}")
        #print(filtProj)
        ifft_filtProj = torch.fft.ifft(filtProj)
        
        filtSino[:,i] = torch.real(ifft_filtProj)

    return filtSino.T

def sample_handler_gpu(measurements_path, shape_path, num_angles_limit=(20,120)):
    """ Uses the pytorch implementation of the backprojection
    """
    #measurements = np.load(measurements_path)
    shape = np.load(shape_path)

    # Randomly choose angles
    num_angles = np.random.randint(num_angles_limit[0], num_angles_limit[1])
    angles = np.random.choice(np.arange(0,180), num_angles, replace=False)

    radon = Radon(shape.shape[0], angles, clip_to_circle=True)
    shape_pt = torch.tensor(shape, dtype=torch.float32, device='cuda')
    measurements = radon.forward(shape_pt)
    
    outer_mask = get_outer_mask(Circle.from_matrix(shape), angles, zero_threshold=0.1)
    
    # Convert to PyTorch tensors
    #measurements = torch.tensor(measurements, dtype=torch.float32, device='cuda')
    measurements = filter_sinogram(measurements)
    # Reconstruct the shape
    reconstruction = radon.backprojection(measurements)
    #print(f"Shapes: Reconstruction: {reconstruction.shape}, outer_mask: {outer_mask.shape}")
    # Clip outer mask and reconstruction to 128x128
    outer_mask = outer_mask[:128,:128]
    reconstruction = reconstruction[:128,:128]
    #print(f"Shapes: Reconstruction: {reconstruction.shape}, outer_mask: {outer_mask.shape}")
    # Scale the reconstruction to be between 0 and 1
    reconstruction = (reconstruction - torch.min(reconstruction)) / (torch.max(reconstruction) - torch.min(reconstruction))
    reconstruction = reconstruction * outer_mask
    reconstruction = (reconstruction - torch.min(reconstruction)) / (torch.max(reconstruction) - torch.min(reconstruction))
    
    # Return the reconstructed shape and the true shape
    shape = torch.tensor(shape, dtype=torch.float32, device='cuda')
    reconstruction = reconstruction.unsqueeze(0)
    shape = shape.unsqueeze(0)
    #print(f"Handler: Reconstruction: {reconstruction.shape}, shape: {shape.shape}")
    return reconstruction, shape
    


class MyDataset(Dataset):
    def __init__(self, folder_path, num_angles_limit=(30, 120), cache_prefix='cached_'):
        self.folder_path = folder_path
        self.num_angles_limit = num_angles_limit
        self.file_list = self._get_file_list()
        self.cache_prefix = cache_prefix

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        cached_data = self.load_from_cache(index)
        if cached_data is not None:
            return cached_data
        file_name = self.file_list[index]
        measurements_path = os.path.join(self.folder_path, file_name)
        shape_path = os.path.join(self.folder_path, file_name.replace('measurements', 'shape'))
        reconstruction, shape = sample_handler_gpu(measurements_path, shape_path, self.num_angles_limit)
        # Cache the result by saving to a numpy file
        np.save(f"{self.folder_path}/{self.cache_prefix}{index}.npy", np.concatenate([reconstruction.cpu().numpy(), shape.cpu().numpy()], axis=0))
        return reconstruction, shape
    
    def load_from_cache(self, index):
        try:
            cached_data = np.load(f"{self.folder_path}/{self.cache_prefix}{index}.npy")
        except:
            #print(f"Could not file {self.folder_path}/{self.cache_prefix}{index}.npy")
            return None
        reconstruction = torch.tensor(cached_data[0], dtype=torch.float32, device='cuda')
        shape = torch.tensor(cached_data[1], dtype=torch.float32, device='cuda')
        reconstruction = reconstruction.unsqueeze(0)
        shape = shape.unsqueeze(0)
        #print(f"Loaded from cache: Reconstruction: {reconstruction.shape}, shape: {shape.shape}")
        return reconstruction, shape

    def _get_file_list(self):
        file_list = []
        for file_name in os.listdir(self.folder_path):
            if file_name.startswith('measurements') and file_name.endswith('.npy'):
                file_list.append(file_name)
        return file_list

def train_model(model, dataset, num_epochs=10, batch_size=64, patience=3, validation_split=0.2):
    """
    Train the model on the dataset
    """
    # Split the dataset into training and validation
    num_val = int(len(dataset) * validation_split)
    num_train = len(dataset) - num_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (reconstruction, shape) in enumerate(train_loader):
            #print(f"Shapes: Reconstruction: {reconstruction.shape}, shape: {shape.shape}")
            optimizer.zero_grad()
            y_hat = model(reconstruction)
            y_hat = y_hat.squeeze()
            shape = shape.squeeze()
            loss = criterion(y_hat, shape)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch}, iteration {i}, loss: {loss.item()}")
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, (reconstruction, shape) in enumerate(val_loader):
                y_hat = model(reconstruction)
                y_hat = y_hat.squeeze()
                shape = shape.squeeze()
                loss = criterion(y_hat, shape)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Validation loss: {val_loss}")
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    return model

def show_preds(model, folder, num_preds=5):
    """
    Show some predictions
    """
    dataset = MyDataset(folder)
    for i in range(num_preds):
        idx = np.random.randint(0, len(dataset))
        reconstruction, shape = dataset[idx]
        print(f"Shapes: Reconstruction: {reconstruction.shape}, shape: {shape.shape}")
        reconstruction = reconstruction.unsqueeze(0)
        y_hat = model(reconstruction)
        y_hat = y_hat.squeeze()
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(reconstruction.cpu().numpy().squeeze())
        ax[1].imshow(shape.cpu().numpy().squeeze())
        ax[2].imshow(y_hat.detach().cpu().numpy().squeeze())
        plt.show()

if __name__ == '__main__':

    # Load the model and make a few predictions
    model = UNet()
    model.load_state_dict(torch.load('model_gpu.pt'))
    model = model.to('cuda')
    show_preds(model, 'Circles128x128_1000',3)
    exit()
    
    n = np.random.randint(0, 100)
    measurements_path = f'Circles128x128_1000/measurements_{n}.npy'
    shape_path = f'Circles128x128_1000/shape_{n}.npy'
    rec, shape = sample_handler_gpu(measurements_path, shape_path, num_angles_limit=(130,180))
    # Plot
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(rec.cpu().numpy().squeeze())
    ax[1].imshow(shape.cpu().numpy().squeeze())
    plt.show()
    #exit()
    plt.cla()
    plt.clf()
    plt.close()
    
    # Create the model
    model = UNet()
    model = model.to('cuda')
    
    # Create the dataset
    dataset = MyDataset('Circles128x128_1000', num_angles_limit=(10,90), cache_prefix='cached_gpu_')
    
    # Train the model
    model = train_model(model, dataset, num_epochs=30, batch_size=32, patience=4)
    
    # Show some predictions
    
    # Save the model
    torch.save(model.state_dict(), 'model_gpu.pt')
    