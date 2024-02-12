import numpy as np
import cv2


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