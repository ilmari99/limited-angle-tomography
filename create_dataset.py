import os
import numpy as np
import matplotlib.pyplot as plt

from AbsorptionMatrices.Circle import Circle


def create_dataset(folder,
                   radius,
                   n_samples,
                   nhole_bounds,
                   portion_missing_pixels_bounds = (0.05,0.5),
                   hole_ratio_limit = 10,
                   hole_size_volatility = 0.1,
                   ):
    """ Creates a dataset of circles with holes.
    """
    # Create a circle
    circle = Circle(radius)
    # Get the measurements at 0 degrees
    base_thickness_measurements = circle.get_measurement(0)
    # Get the angles to rotate the circle by
    angle_start = 0
    angle_end = 360
    angle_step = 1
    angles = np.arange(angle_start,angle_end,angle_step)
    os.makedirs(folder,exist_ok=True)
    
    circle_mask = circle.matrix > 0
    
    num_circle_pixels = np.sum(circle_mask)
    # Choose a portion of pixels to be missing
    get_portion_missing_pixels = lambda : np.random.uniform(*portion_missing_pixels_bounds)
    # Choose a number of holes to make
    get_nholes = lambda : np.random.randint(nhole_bounds[0],nhole_bounds[1])
    
    if isinstance(hole_size_volatility,type(callable)):
        hole_size_volatility_fun = hole_size_volatility
    elif isinstance(hole_size_volatility,(float,int)):
        hole_size_volatility_fun = lambda : hole_size_volatility()
    elif isinstance(hole_size_volatility,(tuple,list)):
        hole_size_volatility_fun = lambda : np.random.uniform(*hole_size_volatility)
    else:
        raise TypeError("hole_size_volatility must be a float, int, tuple, list or callable.")
        
    
    # Create the dataset
    for i in range(n_samples):
        
        # Create a new circle instance from the base circle
        temp_circle = Circle.from_matrix(circle.matrix.copy())
        
        p_missing = get_portion_missing_pixels()
        nholes = get_nholes()
        print(f"Making {nholes} holes, with {p_missing} of pixels missing ({int(p_missing * num_circle_pixels)} pixels)")
        try:
            # Make the holes
            temp_circle.make_holes(nholes, int(p_missing * num_circle_pixels), inplace = True, hole_size_volatility=hole_size_volatility_fun(), hole_ratio_limit=hole_ratio_limit)
        except ValueError as e:
            print(f"Could not make holes: {e}")
            continue
        
        # Get the measurements at all angles
        measurements = np.array([temp_circle.get_measurement(theta) for theta in angles])
        # nangles x image_height
        # Adjust the measurements so that the measurements correspond to the difference in thickness from the base thickness
        measurements = base_thickness_measurements - measurements
        
        # Store the measurements and the circle matrix
        np.save(folder + f"/shape_{i}",temp_circle.matrix)
        np.save(folder + f"/measurements_{i}",measurements)

    return


if __name__ == "__main__":
    circle_radius = 40
    
    angle_start = 0
    angle_end = 360
    angle_step = 1
    folder = f"Circle_radius_{circle_radius}_scans"
    
    create_dataset(folder,
                   circle_radius,
                   5000,
                   nhole_bounds=(1,12),
                   portion_missing_pixels_bounds=(0.05,0.5),
                   hole_ratio_limit=10,
                   hole_size_volatility=0.8
                   )