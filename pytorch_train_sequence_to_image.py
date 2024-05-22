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
#from pytorch_models import SequenceToImageCNN

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
    shape = np.load(shape_path)
    
    #print(f"Shape: {shape.shape}")
    # Randomly choose angles
    num_angles = np.random.randint(num_angles_limit[0], num_angles_limit[1])
    #step = np.random.choice([0.5, 1.0])
    angles = np.arange(0,num_angles,1)
    #print(f"Num angles: {num_angles}, angles: {angles}")
    radon = Radon(shape.shape[0], angles, clip_to_circle=True)
    shape_pt = torch.tensor(shape, device='cuda', dtype=torch.float32)
    sinogram = radon.forward(shape_pt)
    
    #outer_mask = get_outer_mask(Circle.from_matrix(shape), angles, zero_threshold=0.1)
    #shape = shape * outer_mask
    
    # Convert to PyTorch tensors
    #sinogram = filter_sinogram(sinogram)
    
    # Return the reconstructed shape and the true shape
    shape = torch.tensor(shape, dtype=torch.float32, device='cuda')
    #shape = shape.unsqueeze(0)
    #print(f"Handler: Sinogram: {sinogram.shape}, shape: {shape.shape}")
    return sinogram, shape
    


class MyDataset(Dataset):
    def __init__(self, folder_path, num_angles_limit=(30, 120), cache_prefix='cached_', sino_shape=(1,180,128), shape_shape=(1,128,128)):
        self.folder_path = folder_path
        self.num_angles_limit = num_angles_limit
        self.file_list = self._get_file_list()
        self.cache_prefix = cache_prefix
        self.sino_shape = sino_shape
        self.shape_shape = shape_shape
        
    def delete_cache(self):
        for file_name in os.listdir(self.folder_path):
            if file_name.startswith(self.cache_prefix):
                os.remove(os.path.join(self.folder_path, file_name))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        cached_data = self.load_from_cache(index)
        if cached_data is not None:
            sinogram, shape = cached_data
            sinogram = torch.nn.functional.pad(sinogram, (0,0,0,self.sino_shape[1] - sinogram.shape[0]))
            #print(f"GetItem (cached): Sinogram: {sinogram.shape}, shape: {shape.shape}")
            return sinogram, shape
        file_name = self.file_list[index]
        measurements_path = os.path.join(self.folder_path, file_name)
        shape_path = os.path.join(self.folder_path, file_name.replace('measurements', 'shape'))
        sinogram, shape = sample_handler_gpu(measurements_path, shape_path, self.num_angles_limit)
        # Cache the result by saving to a numpy file
        np.save(f"{self.folder_path}/{self.cache_prefix}{index}.npy", np.concatenate([sinogram.cpu().numpy(), shape.cpu().numpy()], axis=0))
        # Pad to the missing angles by adding zero rows
        sinogram = torch.nn.functional.pad(sinogram, (0,0,0,self.sino_shape[1] - sinogram.shape[0]),mode='constant', value=0)
        #print(f"GetItem: Sinogram: {sinogram.shape}, shape: {shape.shape}")
        return sinogram, shape
    
    def load_from_cache(self, index):
        try:
            cached_data = np.load(f"{self.folder_path}/{self.cache_prefix}{index}.npy")
        except:
            #print(f"Could not file {self.folder_path}/{self.cache_prefix}{index}.npy")
            return None
        #print(f"Loaded shape: {cached_data.shape}")
        # Shape is the last 128x128 matrix
        shape = torch.tensor(cached_data[-self.shape_shape[1]:], dtype=torch.float32, device='cuda')
        # Sinogram is the first values, before the shape
        sinogram = torch.tensor(cached_data[:-self.shape_shape[1]], dtype=torch.float32, device='cuda')
        return sinogram, shape

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
        dataset.delete_cache()
        for i, (sinogram, shape) in enumerate(train_loader):
            #print(f"Shapes: sinogram: {sinogram.shape}, shape: {shape.shape}")
            optimizer.zero_grad()
            y_hat = model(sinogram)
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
            for i, (sinogram, shape) in enumerate(val_loader):
                y_hat = model(sinogram)
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

class SequenceToImageCNN(nn.Module):
    """ An encode-decoder CNN that takes in a sinogram (D x N) and outputs an image (N x N).
    Each row in the sinogram is a projection (1xN), and the number of projections is D.
    The encoder is an LSTM that takes in the sinogram, and the decoder is a CNN that takes in the state of the LSTM after
    all projections have been fed in.
    """
    def __init__(self, input_shape, output_shape, hidden_size=128, num_layers=1):
        super(SequenceToImageCNN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.encoder = nn.LSTM(input_size=input_shape[2], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Now, the encoder will take in a sinogram, and output a hidden state containing the information of the sinogram
        # The decoder will take in the hidden state, and output a vector with prod(output_shape) elements
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, np.prod(output_shape)),
            nn.Sigmoid()
        )

        
    def forward(self, s):
        #print(f"Forward: Sinogram: {s.shape}")
        # Pad the sinograms to have input_shape[1] rows
        #s = s.squeeze(0)
        #s = torch.nn.functional.pad(s, (0, 0, 0, self.input_shape[1] - s.shape[0]))
        s = s.reshape((-1, self.input_shape[1], self.input_shape[2]))
        #print(f"Forward: Sinogram: {s.shape}")
        # Pass the sinogram through the LSTM, and get the hidden state
        _, (h_n, c_n) = self.encoder(s)
        # Pass the hidden state through the decoder
        dec = self.decoder(h_n)
        #print(f"Forward: Dec: {dec.shape}")
        dec = dec.reshape((-1,self.output_shape[0], self.output_shape[1]))
        #print(f"Forward: Dec: {dec.shape}")
        return dec

def show_preds(model, folder, num_preds=5):
    """
    Show some predictions
    """
    dataset = MyDataset(folder, cache_prefix="cached_sino_gpu_",sino_shape=(1,180,128), shape_shape=(1,128,128), num_angles_limit=(160,180))
    dataset.delete_cache()
    for i in range(num_preds):
        idx = np.random.randint(0, len(dataset))
        sinogram, shape = dataset[idx]
        print(f"Sinogram: {sinogram.shape}, shape: {shape.shape}")
        y_hat = model(sinogram.unsqueeze(0))
        y_hat = y_hat.squeeze()
        shape = shape.squeeze()
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(sinogram.cpu().numpy().squeeze().T)
        ax[0].set_title('Sinogram')
        ax[1].imshow(y_hat.cpu().detach().numpy().squeeze())
        ax[1].set_title('Prediction')
        ax[2].imshow(shape.cpu().numpy().squeeze())
        ax[2].set_title('True shape')
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        

if __name__ == '__main__':

    if True:
        # Load the model and make a few predictions
        model = SequenceToImageCNN(input_shape=(1, 180, 128), output_shape=(128, 128), hidden_size=2056)
        model.load_state_dict(torch.load('model_Seq2Img.pt'))
        model = model.to('cuda')
        show_preds(model, 'Circles128x128_1000',3)
        exit()
    
    # Create the model
    model = SequenceToImageCNN(input_shape=(1, 180, 128), output_shape=(128, 128), hidden_size=2056, num_layers=1)
    model = model.to('cuda')
    
    # Create the dataset
    dataset = MyDataset('Circles128x128_1000', num_angles_limit=(10,180), cache_prefix="cached_sino_gpu_", sino_shape=(1,180,128), shape_shape=(1,128,128))
    dataset.delete_cache()
    
    # Train the model
    model = train_model(model, dataset, num_epochs=60, batch_size=32, patience=4)
    
    # Show some predictions
    
    # Save the model
    torch.save(model.state_dict(), 'model_Seq2Img.pt')
    