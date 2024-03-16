import numpy as np
import matplotlib.pyplot as plt
from AbsorptionMatrices import Full, Circle
from reconstruct import (backproject_with_distance_measures,
                         calc_masks_from_distances,
                         filter_sinogram)

"""
This is a short script I started doing to attempt to model where are holes
in an object by rotating it and seeing how much radiation is absorbed in which part.
"""

def reconstruct_error(m, m_hat):
    """ Calculate the l1 norm
    """
    return np.linalg.norm(m - m_hat, ord=1)

if __name__ == "__main__":
    
    # Get a shape
    def get_shape():
        return Circle(64)
    
    shape = get_shape()

    shape.make_holes(10, 0.3, hole_size_volatility=0.2, at_angle=0)
    
    # Plot the shape
    fig,ax = shape.plot()
    ax.set_title("Ground truth shape of the object")
    
    
    ZERO_THRESHOLD = 0.1
    
    # At which angles to take the measurements
    angles = np.arange(0,180,1)
    
    # Measure the backprojections (i.e. the CT scans) at each angle
    # Also measure the distances from the front and back of the object at each angle.
    # This can for example be obtained by laser scanning the object while taking the CT scans.
    (sinogram,
     distances_from_front,
     distances_from_back) = shape.get_multiple_measurements(
                            angles,
                            return_distances=True,
                            zero_threshold=ZERO_THRESHOLD
                            )
     
    # Based on distances from front and back, we can calculate the thickness of the object at each height,
    # and adjust the sinogram
    thicknesses = np.full(sinogram.shape, sinogram.shape[1])
    thicknesses = thicknesses - distances_from_front - distances_from_back
    # Now, to adjust the measurements we can calculate the number of missing pixels at each height
    # which is
    adjusted_sinogram = thicknesses - sinogram
    
    # Plot the sinogram, adjusted sinogram, and the filtered sinogram
    fig, ax = plt.subplots(1,3)
    fig.suptitle(f"Reconstruction using {len(angles)} degrees")
    ax[0].imshow(sinogram.T)
    ax[0].set_title("Sinogram")
    ax[1].imshow(adjusted_sinogram.T)
    ax[1].set_title("Sinogram adjusted on the thickness of the object")
    ax[2].imshow(filter_sinogram(adjusted_sinogram, a=0.1).T)
    ax[2].set_title("Adjusted and filtered sinogram")

    # Calcualate the outer mask of the object
    masks = calc_masks_from_distances(distances_from_front, distances_from_back)
    f = Full(size = (sinogram.shape[1],sinogram.shape[1]))
    for angle, mask in zip(angles, masks):
        f.rotate(angle, inplace=True)
        f.matrix *= mask
        f.rotate(-angle, inplace=True)
    outer_mask = f.matrix > ZERO_THRESHOLD
    fig, ax = plt.subplots()
    ax.matshow(outer_mask)
    ax.set_title("Topology of the objects outer surface")
        
    
    # Reconstruct the shape
    reconstr_shape = backproject_with_distance_measures(adjusted_sinogram,
                                                                 angles,
                                                                 distances_from_front,
                                                                 distances_from_back,
                                                                 use_filter=False,
                                                                 zero_threshold=ZERO_THRESHOLD
                                                                 )
    
    fig, ax = reconstr_shape.plot()
    fig.set_size_inches(10,10)
    err = reconstruct_error(shape.matrix, reconstr_shape.matrix)
    squared_error = np.mean((shape.matrix - reconstr_shape.matrix)**2).round(3)
    fig.suptitle(f"Reconstruction using {len(angles)} degrees")
    ax.set_title(f"Reconstructed object using backprojection with adjusted sinogram.\nMSE: {squared_error}")
    
    fbp_reconstr = backproject_with_distance_measures(adjusted_sinogram,
                                                                angles,
                                                                distances_from_front,
                                                                distances_from_back,
                                                                use_filter=True,
                                                                zero_threshold=ZERO_THRESHOLD
                                                                )
    
    fig, ax = fbp_reconstr.plot()
    fig.set_size_inches(10,10)
    err = reconstruct_error(shape.matrix, fbp_reconstr.matrix)
    squared_error = np.mean((shape.matrix - fbp_reconstr.matrix)**2).round(3)
    fig.suptitle(f"Reconstruction using {len(angles)} degrees")
    ax.set_title(f"Reconstructed object using Filtered Back projection with adjusted sinogram.\nMSE: {squared_error}")
    
    plt.show()