from AbsorptionMatrices import Square, Full
from AbsorptionMatrices.AbsorptionMatrix import AbsorptionMatrix
import numpy as np
from numpy.fft import fftshift, fft, ifft


def backproject_to_large_square(sinogram, angles, clip_to_original_size = False, angle_masks = None, use_filter = False):
    """ With this reconstruction method, we create a square as the base shape, unto which we rollback the sinogram.
    We hence require, that the side length of the square is equal to the full image,
    and that the side_length of the absorption martix is sqrt(2) times the side length (to allow for full rotation).
    """
    side_len = int(sinogram.shape[1])
    print(f"Side len: {side_len}")
    shape = Square(side_len)
    print(f"Shape size: {shape.size}")
    reconstr_shape = backproject_to_shape(sinogram, angles, shape, angle_masks, use_filter)
    if clip_to_original_size:
        reconstr_shape = clip_to_original(shape, reconstr_shape)
        # Re scale
        scaler = lambda x : (x - np.min(x)) / (np.max(x) - np.min(x))
        reconstr_shape.matrix = scaler(reconstr_shape.matrix)
    return reconstr_shape


def filter_sinogram(sino, a = 0.1):
    """filter projections. Normally a ramp filter multiplied by a window function is used in filtered
    backprojection. The filter function here can be adjusted by a single parameter 'a' to either approximate
    a pure ramp filter (a ~ 0)  or one that is multiplied by a sinc window with increasing cutoff frequency (a ~ 1).
    Credit goes to Wakas Aqram. 
    inputs: sino - [n x m] numpy array where n is the number of projections and m is the number of angles used.
    outputs: filtSino - [n x m] filtered sinogram array
    
    Reference: https://github.com/csheaff/filt-back-proj
    """
    sino = sino.T
    projLen, numAngles = sino.shape
    step = 2*np.pi/projLen
    w = np.arange(-np.pi, np.pi, step)
    if len(w) < projLen:
        w = np.concatenate([w, [w[-1]+step]]) #depending on image size, it might be that len(w) =  
                                              #projLen - 1. Another element is added to w in this case
    rn1 = abs(2/a*np.sin(a*w/2));  #approximation of ramp filter abs(w) with a funciton abs(sin(w))
    rn2 = np.sin(a*w/2)/(a*w/2);   #sinc window with 'a' modifying the cutoff freqs
    r = rn1*(rn2)**2;              #modulation of ramp filter with sinc window
    
    filt = fftshift(r)   
    filtSino = np.zeros((projLen, numAngles))
    for i in range(numAngles):
        projfft = fft(sino[:,i])
        filtProj = projfft*filt
        filtSino[:,i] = np.real(ifft(filtProj))

    return filtSino.T
    
    
    
def clip_to_original(shape, reconstr_shape):
    """ Only take the middle part shape that has the same size as the original shape.
    Typically called after reconstructing a shape to a bigger square.
    
    We clip the matrix by finding the outermost non-zero elements in each direction,
    and then we pad the matrix so that it is square.
    """
    
    # Cut the matrix to the original size
    assert isinstance(shape, Square), "Only squares are supported."
    side_len = shape.side_len
    leftmost_col = 0
    rightmost_col = reconstr_shape.matrix.shape[1]
    uppermost_row = 0
    lowermost_row = reconstr_shape.matrix.shape[0]
    for col in range(reconstr_shape.matrix.shape[1]):
        if np.sum(reconstr_shape.matrix[:,col]) > 0:
            leftmost_col = col
            break
    for col in range(reconstr_shape.matrix.shape[1]-1,-1,-1):
        if np.sum(reconstr_shape.matrix[:,col]) > 0:
            rightmost_col = col
            break
    for row in range(reconstr_shape.matrix.shape[0]):
        if np.sum(reconstr_shape.matrix[row,:]) > 0:
            uppermost_row = row
            break
    for row in range(reconstr_shape.matrix.shape[0]-1,-1,-1):
        if np.sum(reconstr_shape.matrix[row,:]) > 0:
            lowermost_row = row
            break
    reconstr_mat = reconstr_shape.matrix[uppermost_row:lowermost_row,leftmost_col:rightmost_col]
    # Pad so that the matrix is square
    if reconstr_mat.shape[0] < side_len:
        pad = side_len - reconstr_mat.shape[0]
        print(f"Padding: {pad} to the bottom")
        reconstr_mat = np.pad(reconstr_mat, ((0,pad),(0,0)))
    if reconstr_mat.shape[1] < side_len:
        pad = side_len - reconstr_mat.shape[1]
        print(f"Padding: {pad} to the right")
        reconstr_mat = np.pad(reconstr_mat, ((0,0),(0,pad)))
        
    reconstr_shape = type(shape).from_matrix(reconstr_mat)
    return reconstr_shape

def calc_mask_from_distances(distance_from_front, distance_from_back):
    """ Return a mask, where elements between distance_from_front and distance_from_back are 1, and 0 otherwise.
    """
    mask = np.zeros((distance_from_front.shape[0], distance_from_front.shape[0]))
    # For each row, set the elements between distance_from_front and distance_from_back to 1
    for i in range(distance_from_front.shape[0]):
        mask[i,distance_from_front[i]:-distance_from_back[i]-1] = 1
    return mask

def calc_masks_from_distances(distances_from_front, distances_from_back):
    """ Calculate the outer shape mask for each angle, based on it's corresponding distance measures.
    """
    masks = np.zeros((distances_from_front.shape[0], distances_from_front.shape[1], distances_from_front.shape[1]))
    for i in range(distances_from_front.shape[0]):
        mask = calc_mask_from_distances(distances_from_front[i,:], distances_from_back[i,:])
        masks[i,:,:] = mask
    return masks

def backproject_with_distance_measures(sinogram,
                                        angles,
                                        distances_from_front,
                                        distances_from_back,
                                        zero_threshold = 0.01,
                                        use_filter = False,
                                        ):
    """ Reconstruct the object from a sinogram and distance measures (for example using a laser).
    This function also uses the distance measures, to calculate the topology of the surface of the object,
    and uses this mask to help in the reconstruction.
    
    Args:
        - sinogram: The sinogram of the object at different angles (nangles x proj_len)
        - angles: The angles at which the sinogram were taken (nangles)
        - distances_from_front: The distances from the front of the object to the object at each angle (nangles x proj_len)
        - distances_from_back: The distances from the back of the object to the object at each angle (nangles x proj_len)
        - zero_threshold: The threshold for the mask that is used to reconstruct the outer shape of the object.
        - use_filter: Whether to use filtered backprojection or not. If a float is given, it is used as the a parameter in the filter function.
    """
    masks = calc_masks_from_distances(distances_from_front, distances_from_back)
    #print(f"Created masks: {masks.shape}")
    full_shape = Full(size = (sinogram.shape[1],sinogram.shape[1]))
    outer_mask = reconstruct_outer_shape(angles, distances_from_front, distances_from_back, zero_threshold = zero_threshold)
    shape = backproject_to_shape(sinogram, angles, full_shape, angle_masks= masks, use_filter=use_filter)
    shape.matrix *= outer_mask
    return shape

def reconstruct_outer_shape(angles, distances_from_front, distances_from_back, zero_threshold = 0.01):
    """ Reconstruct the outer shape of the object from distance measures.
    So this gives the surface tomography of the object.
    """
    masks = calc_masks_from_distances(distances_from_front, distances_from_back)
    # We create a shape full of ones, and then at each angle, we rotate the shape, and multiply it with the mask,
    # So we end up with a final mask, that is the topology of the object as calculated from the distance measures.
    full_shape = Full(size = (masks[0].shape[0],masks[0].shape[0]))
    #print(f"Full shape size: {full_shape.size}")
    for angle, mask in zip(angles, masks):
        full_shape.rotate(angle, inplace=True)
        full_shape.matrix *= mask
        full_shape.rotate(-angle, inplace=True)
    outer_mask = full_shape.matrix > zero_threshold
    return outer_mask


def backproject_to_shape(sinogram, angles, shape : AbsorptionMatrix, angle_masks = None, use_filter = False):
    """ Backproject the sinogram to a shape in order to reconstruct the object.
    Each row in sinogram is a backprojection of the object at a certain angle.
    
    Args:
        - sinogram: The sinogram of the object at different angles (nangles x proj_len)
        - angles: The angles at which the sinogram were taken (nangles)
        - shape (AbsorptionMatrix): The shape to backproject to.
        - angle_masks: Mask based on the outer shape of the object at each angle. If None, no masks are used.
        - use_filter: Whether to use filtered backprojection or not. If a float is given, it is used as the a parameter in the filter function.
    """
    if use_filter:
        if isinstance(use_filter, float):
            sinogram = filter_sinogram(sinogram, use_filter)
        else:
            sinogram = filter_sinogram(sinogram)

    gt_zero_mask = shape.matrix > 0
    i = 0
    # Pad the sinogram equally on both sides, so that each measurement is the same length as shape.matrix.shape[1]
    total_pad = shape.matrix.shape[1] - sinogram.shape[1]
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    sinogram = np.pad(sinogram, ((0,0),(pad_left,pad_right)))
    for angle, measurement in zip(angles,sinogram):
        # Rotate the matrix to angle
        shape.rotate(angle, inplace=True)
        # For every element in the measurement vector, subtract it from the corresponding row in the reconstruction
        m_matrix = np.repeat(measurement,shape.matrix.shape[1]).reshape(shape.matrix.shape)
        # Subtract the sinogram from the reconstructed circle
        sub = m_matrix
        if angle_masks is not None:
            sub = sub * angle_masks[i]
        shape.matrix -= sub
        # Rotate back
        shape.rotate(-angle, inplace=True)
        i += 1
    
    # Invert the sign of the matrix, and scale to be between 0 and 1
    # Set points outside the circle to be as negative as possible, because we know there is nothing there
    shape.matrix = np.where(gt_zero_mask,shape.matrix,-np.max(np.abs(shape.matrix)))
    # Now, the smalle the values are, the more likely it is that there is nothing there.
    # We want to scale the values to be between 0 and 1, so 0 is surely nothing, and 1 is surely something.
    scaler = lambda x : (x - np.min(x)) / (np.max(x) - np.min(x))
    shape.matrix = scaler(shape.matrix)
    return shape