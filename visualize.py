import os
import numpy as np
import matplotlib.pyplot as plt

def plot_measurements(measurements):
    # So each row contains how much of the object is at each height, for a given angle.
    # Plot, so that the y axis is the height, and the x axis is the angle.
    fig, ax = plt.subplots()
    
    # Show before removing noise
    ax.matshow(measurements.T)
    ax.set_title("Measurements before removing noise")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Height")
    
    return fig,ax

if __name__ == "__main__":
    folder = "/home/ilmari/python/limited-angle-tomography/DatascansTest2"
    measurements = list(filter(lambda x: "measurements" in x, os.listdir(folder)))
    measurements = sorted(measurements, key=lambda x: int(x.split("_")[1].split(".")[0]))
    shapes = list(filter(lambda x: x.startswith("shape"), os.listdir(folder)))
    shapes = sorted(shapes, key=lambda x: int(x.split("_")[1].split(".")[0]))
    reconstructions = list(filter(lambda x: x.startswith("reconstructed"), os.listdir(folder)))
    reconstructions = sorted(reconstructions, key=lambda x: int(x.split("_")[2]))
    print(f"Found {len(measurements)} measurements, {len(shapes)} shapes and {len(reconstructions)} reconstructions")
    # Show first 5
    for i in range(5):
        fig, ax = plt.subplots(1,3)
        # Orig shape
        orig_shape = np.load(os.path.join(folder, shapes[i]))
        ax[0].imshow(orig_shape)
        ax[0].set_title("Original shape")
        # Measurements
        meas = np.load(os.path.join(folder, measurements[i]))
        ax[1].imshow(meas.T)
        ax[1].set_title("Measurements")
        # Reconstructed shape
        reconstr_shape = np.load(os.path.join(folder, reconstructions[i]))
        ax[2].imshow(reconstr_shape)
        ax[2].set_title("Reconstructed shape")
        plt.show()
        
        