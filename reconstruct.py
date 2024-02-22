from AbsorptionMatrices.Circle import Circle
from AbsorptionMatrices.AbsorptionMatrix import AbsorptionMatrix
import numpy as np
from cv2 import morphologyEx, connectedComponents


def rollback_reconstruct_circle(measurements, angles, radius):
    shape = Circle(radius)
    return rollback_reconstruct_shape(measurements, angles, shape)
    
def rollback_reconstruct_circle_angle_step(measurements, angle_step, gt_zero_radius):
    return rollback_reconstruct_circle(measurements, np.arange(0,len(measurements)*angle_step,angle_step), gt_zero_radius)


def rollback_reconstruct_shape(measurements, angles, shape : AbsorptionMatrix):
    # Now the pattern shows multiple sine waves with different thicknesses, and which occassionally cross each other.
    # The pattern shows measurements of a function f(theta) -> R^(image_height), where theta is the angle, and
    # R^(image_height) is the corresponding measurement.
    
    # Each row in measurements is the measurement at the corresponding angle in angles.
    
    # Lets now attempt to reconstruct the circle with holes from the measurements.
    # We do this by 'unrolling'. We start with a circle, rotate it by theta, and then reduce each row by the corresponding
    # measurement at that angle.
    
    # This mask will be 1 where the circle is, and 0 where the circle is not.
    gt_zero_mask = shape.matrix > 0
    i = 0
    # Pad the measurements equally on both sides, so that each measurement is the same length as shape.matrix.shape[1]
    total_pad = shape.matrix.shape[1] - measurements.shape[1]
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    measurements = np.pad(measurements, ((0,0),(pad_left,pad_right)))
    for angle, measurement in zip(angles,measurements):
        # Rotate the matrix to angle
        shape.rotate(angle, inplace=True)
        # For every element in the measurement vector, subtract it from the corresponding row in the reconstruction
        m_matrix = np.repeat(measurement,shape.matrix.shape[1]).reshape(shape.matrix.shape)
        # Subtract the measurements from the reconstructed circle
        shape.matrix -= m_matrix#*gt_zero_mask
        #for i in range(len(measurement)):
        #    shape.matrix[i,:] -= measurement[i]# * gt_zero_mask[i,:]
        # Rotate back
        shape.rotate(-angle, inplace=True)
        i += 1
    
    # Invert the sign of the matrix, and scale to be between 0 and 1
    # Set points outside the circle to be as negative as possible, because we know there is nothing there
    #shape.matrix = np.where(gt_zero_mask,shape.matrix,-np.max(np.abs(shape.matrix)))
    # Now, the smalle the values are, the more likely it is that there is nothing there.
    # We want to scale the values to be between 0 and 1, so 0 is surely nothing, and 1 is surely something.
    scaler = lambda x : (x - np.min(x)) / (np.max(x) - np.min(x))
    shape.matrix = scaler(shape.matrix)
    return shape