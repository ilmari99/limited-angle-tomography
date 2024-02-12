import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from AbsorptionMatrices import Square, AbsorptionMatrix
from visualize import plot_measurements
from utils import remove_noise_from_measurements, reconstruct_error
from reconstruct import rollback_reconstruct_shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Skimage
from skimage import filters
# Canny
from skimage import feature
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import flood, flood_fill

"""
This is a short script I started doing to attempt to model where are holes
in an object by rotating it and seeing how much radiation is absorbed in which part.
"""


if __name__ == "__main__":
    
    get_shape = lambda: Square(100)
    
    shape = get_shape()
    base_shape = get_shape()

    angles = np.arange(0,90,0.2)
    shape.make_holes(10, 0.4, hole_size_volatility=0.1)
    shape.plot()
    
    # The base measurements give us priori information about the shape of the object.
    base_measurements = np.array([base_shape.get_measurement(theta) for theta in angles])
    measurements = np.array([shape.get_measurement(theta) for theta in angles])
    
    number_of_blocks_on_row_h_in_base_shape_during_rotation = np.sum(base_measurements, axis = 0)
    number_of_blocks_on_row_h_in_real_shape_during_rotation = np.sum(measurements, axis = 0)
    number_of_missing_blocks_on_row_h = number_of_blocks_on_row_h_in_base_shape_during_rotation - number_of_blocks_on_row_h_in_real_shape_during_rotation

    # Now, for each column in measurements, we multiply by the corresponding coefficient
    measurements = base_measurements - measurements
    # Divide each value by base_measurements, and scale to 0-1
    measurements = measurements# / base_measurements
    
    
    fig, ax = plt.subplots()
    ax.plot(number_of_blocks_on_row_h_in_base_shape_during_rotation, label = "Base shape")
    ax.plot(number_of_blocks_on_row_h_in_real_shape_during_rotation, label = "Real shape")
    ax.plot(number_of_missing_blocks_on_row_h, label = "Missing blocks")
    ax.set_title("Number of blocks on row h during rotation")
    ax.legend()
    #plt.show()
    #exit()
    
    print(f"Measurements shape: {measurements.shape}")
    
    # Plot the measurements
    fig,ax = plot_measurements(measurements)
    ax.set_title("Measurements before removing noise")
    
    # Remove noise from the measurements
    measurements_no_noise = remove_noise_from_measurements(measurements)
    fig,ax = plot_measurements(measurements_no_noise)
    ax.set_title("Measurements after removing noise")
    
    # Plot the shape
    fig,ax = shape.plot()
    ax.set_title("shape with holes")
    print(f"Measurements shape: {measurements.shape}")
    # Reconstruct the shape
    reconstr_shape = rollback_reconstruct_shape(measurements, angles, base_shape)
    
    fig, ax = reconstr_shape.plot()
    err = reconstruct_error(shape.matrix, reconstr_shape.matrix)
    squared_error = np.sum((shape.matrix - reconstr_shape.matrix)**2)
    ax.set_title(f"Reconstructed shape with holes. Error: {squared_error}")
    
    
    # Canny
    edges = feature.canny(reconstr_shape.matrix)
    
    # Connect the edges
    edges = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
    # Find the largest pixel value in reconstr_shape.matrix, that is 0 in edges
    m1 = edges == 0
    m2 = base_shape.matrix > 0
    largest_value = np.max(reconstr_shape.matrix[m1 & m2])
    print(f"Largest value: {largest_value}")
    selected_pixel = np.argwhere(reconstr_shape.matrix == largest_value)[0]
    print(f"In pixel {selected_pixel}")
    #print(f"Value in edges: {edges[selected_pixel[0], selected_pixel[1]]}")
    
    reconstr = flood_fill(edges, tuple(selected_pixel), 1, connectivity=1)
    fig, ax = plt.subplots()
    ax.matshow(reconstr)
    err = reconstruct_error(shape.matrix, reconstr)
    squared_error = np.sum((shape.matrix - reconstr)**2)
    ax.set_title(f"Enhanced reconstruction using Canny. Error: {squared_error}")
    
    plt.show()