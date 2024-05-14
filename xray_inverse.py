import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import skimage
from AbsorptionMatrices import Square, AbsorptionMatrix, Full, FilledBezierCurve, Circle
from visualize import plot_measurements
from utils import remove_noise_from_measurements, reconstruct_error
from reconstruct import (backproject_to_shape,
                         backproject_to_large_square,
                         backproject_with_distance_measures,
                         calc_mask_from_distances,
                         calc_masks_from_distances,
                         filter_sinogram)
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
    
    def get_shape():
        return Square(70)
        success = False
        shape = None
        while not success:
            shape = FilledBezierCurve((160,160), shape_size_frac=0.8, on_fail="ignore", n_bezier_points=12)
            success = shape.SUCCESS
        return shape
    
    shape = get_shape()
    base_shape = get_shape()

    angles = np.arange(0,180,1)
    shape.make_holes(10, 0.6, hole_size_volatility=0.2, at_angle=20)
    ZERO_THRESHOLD = 0.1
    
    # The base measurements give us priori information about the shape of the object.
    (measurements,
     distances_from_front,
     distances_from_back) = shape.get_multiple_measurements(
                            angles,
                            return_distances=True,
                            zero_threshold=ZERO_THRESHOLD
                            )
     
    sinogram = skimage.transform.radon(shape.matrix.T, theta=angles, circle=True)
    
    fig, ax = plt.subplots()
    ax.matshow(sinogram)
    ax.set_title("Sinogram")
    
    fig, ax = plt.subplots()
    ax.matshow(measurements)
    ax.set_title("Measurements")
    
        
    # Based on distances from front and back, we can calculate the thickness of the object at each height.
    thicknesses = np.full(measurements.shape, measurements.shape[1])
    thicknesses = thicknesses - distances_from_front - distances_from_back
    #print(f"Thickenesses: {thicknesses}")
    # Now, to adjust the measurements we can calculate the number of missing pixels at each height
    # which is
    adjusted_measurements = thicknesses - measurements

    if True:
        masks = calc_masks_from_distances(distances_from_front, distances_from_back)
        if False:
            # Animate the masks
            fig, ax = plt.subplots()
            for i in range(masks.shape[0]):
                ax.matshow(masks[i,:,:])
                plt.pause(0.01)
                ax.clear()
        
        mask0 = calc_mask_from_distances(distances_from_front[0,:], distances_from_back[0,:])
        fig, ax = plt.subplots()
        ax.matshow(mask0)
        ax.set_title("Distance mask at angle 0")
    
        f = Full(size = (measurements.shape[1],measurements.shape[1]))
        for angle, mask in zip(angles, masks):
            f.rotate(angle, inplace=True)
            f.matrix *= mask
            f.rotate(-angle, inplace=True)
        outer_mask = f.matrix > ZERO_THRESHOLD
        fig, ax = plt.subplots()
        ax.matshow(outer_mask)
        ax.set_title("Outer mask")
    
    # Plot the shape
    fig,ax = shape.plot()
    ax.set_title("shape with holes")
    print(f"Measurements shape: {adjusted_measurements.shape}")
    
    # Reconstruct the shape
    reconstr_shape = backproject_with_distance_measures(adjusted_measurements,
                                                                 angles,
                                                                 distances_from_front,
                                                                 distances_from_back,
                                                                 use_filter=False,
                                                                 zero_threshold=ZERO_THRESHOLD
                                                                 )

    fig, ax = reconstr_shape.plot()
    err = reconstruct_error(shape.matrix, reconstr_shape.matrix)
    squared_error = np.sum((shape.matrix - reconstr_shape.matrix)**2)
    ax.set_title(f"Reconstructed shape with holes. Error: {squared_error}")
    
    fbp_reconstr = backproject_with_distance_measures(adjusted_measurements,
                                                                angles,
                                                                distances_from_front,
                                                                distances_from_back,
                                                                use_filter=True,
                                                                zero_threshold=ZERO_THRESHOLD
                                                                )
    
    fig, ax = fbp_reconstr.plot()
    err = reconstruct_error(shape.matrix, fbp_reconstr.matrix)
    squared_error = np.sum((shape.matrix - fbp_reconstr.matrix)**2)
    ax.set_title(f"FBP reconstructed shape with holes. Error: {squared_error}")
    
    
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