import numpy as np
import cv2
import torch
from torch_radon import Radon


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
        
class FBPRadon(Radon):
    def __init__(self, resolution, angles, a = 0.1, device='cuda', **kwargs):
        self.device = device
        super(FBPRadon, self).__init__(resolution=resolution, angles=angles, **kwargs)
        self.a = a
        #self.a = torch.tensor(a, device=device, dtype=torch.float32, requires_grad=True)
        #self.parameters = [self.a]
        
    def forward(self, x):
        s = super().forward(x)
        if not self.a:
            return s
        s = filter_sinogram(s, a = self.a, device=self.device)
        return s

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