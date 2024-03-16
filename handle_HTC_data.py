import os
import numpy as np
import matplotlib.pyplot as plt

from reconstruct import backproject_to_large_square

base_path = "/home/ilmari/python/limited-angle-tomography/HTC_files/"
htc_file = "td"
sinogram_file = htc_file + "_sinogram.csv"
angles_file = htc_file + "_angles.csv"

calibration_disk_path = "/home/ilmari/python/limited-angle-tomography/HTC_files/solid_sinogram.csv"
calibration_disk_angles_path = "/home/ilmari/python/limited-angle-tomography/HTC_files/solid_angles.csv"

calibration_sinogram = np.loadtxt(calibration_disk_path, delimiter=",")
calibration_angles = np.loadtxt(calibration_disk_angles_path, delimiter=",")
# Load the sinogram
sinogram = np.loadtxt(os.path.join(base_path, sinogram_file), delimiter=",")
# Load the angles
angles = np.loadtxt(os.path.join(base_path, angles_file), delimiter=",")

assert np.array_equal(angles, calibration_angles), "The angles in the sinogram and the calibration sinogram do not match."

sinogram = calibration_sinogram - sinogram

# Reconstruction
print(f"Sino shape: {sinogram.shape}")

# Add one column of zeros to the sinogram
sinogram = np.hstack((sinogram, np.zeros((sinogram.shape[0], 1))))


selected_angles = np.arange(0, 90, 1)
angles = angles[selected_angles]
sinogram = sinogram[selected_angles]


# Plot the sinogram
fig, ax = plt.subplots()
ax.imshow(sinogram.T)
ax.set_title("Sinogram")
plt.show(block=False)


reconstr_circle = backproject_to_large_square(sinogram, angles, clip_to_original_size=True, use_filter=True)
fig, ax = reconstr_circle.plot()
ax.set_title("Reconstruction")

plt.show()

