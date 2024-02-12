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