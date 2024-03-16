import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle
import tqdm

from AbsorptionMatrices import Circle, Square, AbsorptionMatrix
from reconstruct import backproject_with_distance_measures

def make_sample(nholes,
                n_missing_pixels,
                folder,
                i,
                hole_size_volatility=0.1,
                hole_ratio_limit=10,
                hole_angle=0,
                zero_threshold=0.1):
    """ Make a sample with holes
    """
    shape = shape_maker()

    # Make the holes
    shape.make_holes(nholes, n_missing_pixels,
                            inplace = True,
                            hole_size_volatility=hole_size_volatility,
                            hole_ratio_limit=hole_ratio_limit,
                            at_angle=hole_angle,
                        )

    # Get the measurements at all angles
    measurements, distances_from_front, distances_from_back = shape.get_multiple_measurements(
        np.arange(0,360,1),
        return_distances=True,
        zero_threshold=zero_threshold
    )

    # Store the measurements, and the distances
    # Make a 3D array with the measurements and the distances
    measurements = np.expand_dims(measurements, axis=2)
    distances = np.stack([distances_from_front, distances_from_back], axis=2)
    measurements = np.concatenate([measurements, distances], axis=2)
    # Store the measurements and the circle matrix
    np.save(folder + f"/shape_{i}",shape.matrix)
    np.save(folder + f"/measurements_{i}",measurements)
    return

def make_sample_wrapper(args):
    return make_sample(*args)

def create_dataset_parallel(folder,
                     shape_maker,
                     n_samples,
                     nhole_bounds,
                     portion_missing_pixels_bounds = (0.05,0.5),
                     hole_ratio_limit = 10,
                     hole_size_volatility = 0.1,
                     zero_threshold = 0.1,
                     ncpus = 8,
                    ):
    """ Creates a dataset of shapes with parts removed.
    Args:
        - folder: str, the folder to store the smaples in
        - shape_maker: callable, a function that returns a shape.
        - n_samples: int, the number of samples to create
        - nhole_bounds: tuple, min and max number of holes to make
        - portion_missing_pixels_bounds: tuple, min and max portion of pixels to remove. The portion is relative to the size of the matrix,
        NOT to the number of px == 1.
        - hole_ratio_limit: int, the maximum ratio between the width/height of a rectangle hole
        - hole_size_volatility: float, int, tuple, list or callable. The volatility of the hole sizes.
        At 0, the holes are the same size, at 1, the holes vary a lot in size.
        - zero_threshold: float. What value is considered > 0, when finding the distances from the front and back.
        - ncpus: int, the number of cpus to use
    """

    os.makedirs(folder,exist_ok=True)

    # Choose a portion of pixels to be missing
    get_portion_missing_pixels = lambda : np.random.uniform(*portion_missing_pixels_bounds)
    # Choose a number of holes to make
    get_nholes = lambda : np.random.randint(nhole_bounds[0],nhole_bounds[1])
    get_angle = lambda : np.random.randint(0,360)
    
    if isinstance(hole_size_volatility,type(callable)):
        hole_size_volatility_fun = hole_size_volatility
    elif isinstance(hole_size_volatility,(float,int)):
        hole_size_volatility_fun = lambda : hole_size_volatility
    elif isinstance(hole_size_volatility,(tuple,list)):
        hole_size_volatility_fun = lambda : np.random.uniform(*hole_size_volatility)
    else:
        raise TypeError("hole_size_volatility must be a float, int, tuple, list or callable.")
    
    # Make sure, that the object returned by shape_maker is Pickle-able
    shape = shape_maker()
    loaded_shape = None
    try:
        loaded_shape = pickle.loads(pickle.dumps(shape))
        print(f"Shape {loaded_shape} is pickle-able")
    except Exception as e:
        raise ValueError(f"shape_maker must return an object that can be pickled. Got error: {e}")
    assert loaded_shape is not None and loaded_shape == shape, "shape_maker must return an object that can be pickled."
      
    # Generate arguments for the make_sample function
    def generate_args():
        for i in range(n_samples):
            nholes = get_nholes()
            p_missing = get_portion_missing_pixels()
            yield nholes, int(p_missing * shape.matrix.size), folder, i, hole_size_volatility_fun(), hole_ratio_limit, get_angle(), zero_threshold
        return
    
    # Create and save samples in parallel, unordered and lazy
    gen = generate_args()
    with mp.Pool(min(n_samples,ncpus)) as pool:
        for i in tqdm.tqdm(pool.imap_unordered(make_sample_wrapper, gen, chunksize=1), total=n_samples):
            pass
    return

def plot_5_first_samples(folder):
    """ Plot the first 5 samples in a folder.
    """
    measurement_paths = list(filter(lambda x: "measurements" in x, os.listdir(folder)))
    measurement_paths = sorted(measurement_paths, key=lambda x: int(x.split("_")[1].split(".")[0]))
    shapes = list(filter(lambda x: x.startswith("shape"), os.listdir(folder)))
    shapes = sorted(shapes, key=lambda x: int(x.split("_")[1].split(".")[0]))
    print(f"Found {len(measurement_paths)} measurements, {len(shapes)} shapes")
    # Show first 5
    for i in range(5):
        measurements = np.load(os.path.join(folder, measurement_paths[i]))
        distances_from_front = measurements[:,:,1].astype(np.int32)
        distances_from_back = measurements[:,:,2].astype(np.int32)
        measurements = measurements[:,:,0].astype(np.int32)
        
        thicknesses = np.full(measurements.shape, measurements.shape[1])
        thicknesses = thicknesses - distances_from_front - distances_from_back
        adjusted_measurements = thicknesses - measurements
        print(f"Shape of adjusted measurements: {adjusted_measurements.shape}")
        print(f"Shape of distances_from_front: {distances_from_front.shape}")
        print(f"Shape of distances_from_back: {distances_from_back.shape}")
        reconstruction = backproject_with_distance_measures(adjusted_measurements,
                                                                     np.arange(0,360,1),
                                                                     distances_from_front,
                                                                     distances_from_back,
                                                                    zero_threshold=0.1
                                                                     )
        
        fig, ax = plt.subplots(1,3)
        # Orig shape
        orig_shape = np.load(os.path.join(folder, shapes[i]))
        
        ax[0].imshow(orig_shape)
        ax[0].set_title("Original shape")
        # Measurements
        ax[1].imshow(adjusted_measurements.T)
        ax[1].set_title("Measurements")
        # Reconstructed shape
        ax[2].imshow(reconstruction.matrix)
        ax[2].set_title("Reconstructed shape")
        plt.show()
    return
        


if __name__ == "__main__":
    #plot_5_first_samples("BezierTest10")
    #exit()
    
    circle_radius = 64
    
    def shape_maker():
        shp = Circle(circle_radius)
        return shp
    
    # Dataset args
    n_samples = 20
    folder = f"CircleR{circle_radius}_{n_samples}"
    n_hole_bounds = (10,15)
    portion_missing_pixels_bounds = (0.2, 0.3)
    hole_ratio_limit = 5
    hole_size_volatility = 0.5
    zero_threshold = 0.1
    
    create_dataset_parallel(folder,
                    shape_maker,
                    n_samples,
                    n_hole_bounds,
                    portion_missing_pixels_bounds,
                    hole_ratio_limit,
                    hole_size_volatility,
                    zero_threshold=zero_threshold,
                    ncpus=os.cpu_count() - 1,
                    )
    
    
    
    
    