import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from AbsorptionMatrices import Circle, Square, AbsorptionMatrix
from visualize import plot_measurements
from utils import remove_noise_from_measurements, reconstruct_error
from reconstruct import rollback_reconstruct_shape, rollback_reconstruct_circle
from skimage import filters

base_path = "/home/ilmari/python/limited-angle-tomography/HTC_files/"
htc_file = "td"
sinogram_file = htc_file + "_sinogram.csv"
angles_file = htc_file + "_angles.csv"

# Load the sinogram
sinogram = np.loadtxt(os.path.join(base_path, sinogram_file), delimiter=",")
# Load the angles
angles = np.loadtxt(os.path.join(base_path, angles_file), delimiter=",")

# Reconstruction
print(f"Sino shape: {sinogram.shape}")

# Add one column of zeros to the sinogram
sinogram = np.hstack((sinogram, np.zeros((sinogram.shape[0], 1))))

selected_angles = np.arange(0, 180, 1)
angles = angles[selected_angles]
sinogram = sinogram[selected_angles]


# Crop the sinogram by removing the first and last 100 columns
#sinogram = sinogram[:, 100:-100]

#sinogram = -1 * sinogram

#sinogram = remove_noise_from_measurements(sinogram)

# Plot the sinogram
fig, ax = plt.subplots()
ax.imshow(sinogram.T)
ax.set_title("Sinogram")
plt.show(block=False)
#exit()
side_len = sinogram.shape[1] // np.sqrt(2)
print(f"Side len: {side_len}")

shape = Square(side_len)
reconstr_circle = rollback_reconstruct_shape(sinogram, angles, shape)
fig, ax = reconstr_circle.plot()
ax.set_title("Reconstruction")

plt.show()

