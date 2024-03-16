""" With this file, we train a model to take in a blurry image (reconstructed from measures)
and output the real image.
"""
import tensorflow as tf
import os
from reconstruct import backproject_with_distance_measures
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow_models import create_a_U_net_model


# GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Each measurement file contains 360 x image_size[0] x 3 measurements:
#   - The first is the base sinogram,
#   - The second contains the distances from the front of the object to the non-zero pixel at each angle
#   - The third contains the distances from the back of the object to the non-zero pixel at each angle

# Each shape file contains the ground truth image from which the correponding measuremet were taken


# Create a dataset from the paths
def get_tf_dataset(measurement_paths, shape_paths, image_shape, save_reconstructions = True, overwrite_reconstructions = False):
    """ Create a tf.data.Dataset from the measurement and shape paths.
    The dataset will yield tuples of (reconstructed_shape, real_shape, index, angle)
    
    The dataset is created by loading the measurements and reconstructing the true shape using a limited angle of the sinogram.
    If save_reconstructions is True, the reconstructions will be saved to the same folder as the shapes
    
    Args:
        - measurement_paths: List of paths to the measurements
        - shape_paths: List of paths to the shapes
        Note that these need to be in the same order
        - image_shape: The shape of the images
        - save_reconstructions: Whether to save the reconstructions for faster loading next time
        - overwrite_reconstructions: Whether to overwrite the reconstructions if they exist
    """
    
    def generator():
        for measurement_path, shape_path,i in zip(measurement_paths, shape_paths, range(len(measurement_paths))):
            reconstr_shape_matrix = None
            reconstr_angle = None
            
            # Check if there is a reconstruction file for this shape
            if not overwrite_reconstructions:
                folder = os.path.dirname(shape_path)
                # Find all files that start with "shape_<i>_reconstructed"
                reconstructions = [fname for fname in os.listdir(folder) if fname.startswith(f"shape_{i}_reconstructed")]
                if len(reconstructions) > 0:
                    reconstr_shape_matrix = np.load(os.path.join(folder, reconstructions[0]))
                    reconstr_angle = int(reconstructions[0].split("_")[-1].replace(".npy", "").replace("a", ""))
                    #print(f"Reconstruction found for shape {i}")
            
            if reconstr_shape_matrix is not None:
                shape = np.load(shape_path)
                yield reconstr_shape_matrix, shape, i, reconstr_angle
                continue
            
            try:
                shape = np.load(shape_path)
                measurements = np.load(measurement_path)
            except:
                print(f"Could not load {measurement_path} or {shape_path}")
                continue
            
            # Randomly choose a 30 - 180 degree slice
            slice_length = np.random.randint(20, 90)
            start_index = np.random.randint(0, 359 - slice_length)
            end_index = start_index + slice_length
            
            distances_from_front = measurements[start_index:end_index,:,1].astype(np.int32)
            distances_from_back = measurements[start_index:end_index,:,2].astype(np.int32)
            measurements = measurements[start_index:end_index,:,0]
            
            thicknesses = np.full(measurements.shape, measurements.shape[1])
            thicknesses = thicknesses - distances_from_front - distances_from_back
            #print(f"Thickenesses: {thicknesses}")
            # Now, to adjust the measurements we can calculate the number of missing pixels at each height
            # which is
            measurements = thicknesses - measurements
            
            #print(f"Reconstructing shape {i} from {start_index} to {end_index} degrees")
            #print(f"Shape: {shape.shape}, measurements: {measurements.shape}")
            # Reconstruct the shape
            reconstr_shape = backproject_with_distance_measures(measurements,
                                                                         np.arange(start_index, end_index),
                                                                         distances_from_front,
                                                                         distances_from_back,
                                                                         use_filter=True,
                                                                         zero_threshold=0.1
                                                                         )
            reconstr_shape_matrix = reconstr_shape.matrix
            reconstr_angle = slice_length
            # Save the reconstruction
            if save_reconstructions:
                # Save to shape_{i}_reconstructed.npy
                reconstr_path = shape_path.replace(".npy", f"_reconstructed_a{slice_length}.npy")
                np.save(reconstr_path, reconstr_shape.matrix)

            yield reconstr_shape_matrix, shape, i, reconstr_angle
        return
    ds = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.int32, tf.int32),
                                        output_shapes=(tf.TensorShape(image_shape), tf.TensorShape(image_shape), tf.TensorShape(()), tf.TensorShape(())))
    return ds


folder = "./limited-angle-tomography/CircleR64_20"
num_samples = 20
image_shape = (129,129)
overwrite_reconstructions = False
save_reconstructions = True
load_model = "./limited-angle-tomography/CircleR64_2000_model.keras"

basename = os.path.basename(folder)
save_to = os.path.join(folder.replace(basename, ""), f"{basename}_model.keras")
print(f"Saving model to {save_to}")


# There are files named "shape_<n>.npy" and "measurements_<n>.npy" in the folder
# Lets find all of them
measurement_paths = [os.path.join(folder, f"measurements_{i}.npy") for i in range(num_samples)]
shape_paths = [os.path.join(folder, f"shape_{i}.npy") for i in range(num_samples)]

if overwrite_reconstructions:
    ds = get_tf_dataset(measurement_paths, shape_paths, image_shape=image_shape, save_reconstructions=save_reconstructions, overwrite_reconstructions=True)
    # Loop through
    for x,y,i,a in ds:
        pass
    print("Overwritten reconstructions")
    
dataset = get_tf_dataset(measurement_paths, shape_paths, image_shape=image_shape, save_reconstructions=save_reconstructions, overwrite_reconstructions=False)

# Print sizes
first_sample = dataset.take(1)
for x,y,i,a in first_sample:
    print(f"Reconstructed shape: {x.shape}")
    print(f"Real shape: {y.shape}")
    input_shape = x.shape
    
# Wrap dataset to only yield x,y
dataset : tf.data.Dataset = dataset.map(lambda x,y,i,a: (x,y))

# For demo purpose we can use whole data, since we have not trained on this
test_sz = num_samples
train_dataset = test_dataset = dataset

if load_model:
    model = keras.models.load_model(load_model)
else:   
    model = create_a_U_net_model(input_shape=input_shape)

print(model.summary())
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/")

if not load_model:
    test_sz = int(0.2 * num_samples)
    train_dataset = dataset.take(num_samples - test_sz)
    test_dataset = dataset.skip(num_samples - test_sz)
    
    model.fit(train_dataset.batch(32), epochs=50, validation_data=test_dataset.batch(32), callbacks=[early_stop, tensorboard])
    model.save(save_to)

# Visualize reconstructions from the test dataset
random_indices = np.random.randint(0, test_sz, 5)
for i in random_indices:
    # Take a random image from the test dataset
    reconstr_shape, shape = list(test_dataset.skip(i).take(1))[0]
    reconstr_shape = reconstr_shape.numpy()
    shape = shape.numpy()
    # Plot the reconstruction
    fig, ax = plt.subplots(1,3)
    # Plot the real shape
    ax[0].imshow(shape)
    ax[0].set_title("Ground truth object")
    
    ax[1].imshow(reconstr_shape)
    err = np.mean((reconstr_shape - shape)**2)
    mean_bce = np.mean(keras.losses.binary_crossentropy(shape, reconstr_shape))
    ax[1].set_title("Purely algorithmic reconstruction using X degrees of measures")
    ax[1].set_xlabel(f"MSE error: {err :.3f}, BCE error: {mean_bce :.3f}")
    
    # Plot the model's reconstruction
    model_reconstr = model.predict(reconstr_shape.reshape((1,*reconstr_shape.shape)))[0]
    print(f"Model reconstr shape: {model_reconstr.shape}")
    print(f"shape shape: {shape.shape}")
    model_reconstr = model_reconstr.reshape(shape.shape)
    rec_error = np.mean((model_reconstr - shape)**2)
    bse_error = np.mean(keras.losses.binary_crossentropy(shape, model_reconstr))
    ax[2].imshow(model_reconstr)
    ax[2].set_title("U-net reconstruction from the simple reconstruction")
    ax[2].set_xlabel(f"MSE error: {rec_error :.3f}, BCE error: {bse_error :.3f}")
plt.show()
    
    

