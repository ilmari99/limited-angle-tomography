""" With this file, we train a model to take in a blurry image (reconstructed from measures)
and output the real image.
"""
import tensorflow as tf
import os
from AbsorptionMatrices import Square
from reconstruct import backproject_to_shape, backproject_with_distance_measures
import numpy as np
import matplotlib.pyplot as plt
import keras
from utils import reconstruct_error
from tensorflow_models import create_a_U_net_model
import tqdm
from create_dataset import create_dataset

# GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Each measurement contains 360 measurements (one for each angle), and each shape is 64x64
# To train the model, we take measures from  -90 to 90 degrees, and use those to create the first reconstruction
# The model then takes in the reconstruction and outputs the real image (shape_paths)

# Create a dataset from the paths
def get_tf_dataset(measurement_paths, shape_paths, image_shape, save_reconstructions = True, overwrite_reconstructions = False):
    
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


folder = "CircleR64_2000_scans"
num_samples = 1000
image_shape = (200,200)
overwrite_reconstructions = False
save_reconstructions = True

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
# Wrap dataset to only yield x,y
dataset : tf.data.Dataset = dataset.map(lambda x,y,i,a: (x,y))

# Print sizes
first_sample = dataset.take(1)
for x,y in first_sample:
    print(f"Reconstructed shape: {x.shape}")
    print(f"Real shape: {y.shape}")
    input_shape = x.shape

test_sz = int(0.2 * num_samples)

train_dataset = dataset.take(num_samples - test_sz)
test_dataset = dataset.skip(num_samples - test_sz)


#model = create_a_U_net_model(input_shape=input_shape)
model = keras.models.load_model("blurry_to_real_model.keras", custom_objects={"PixelWIseBCE": keras.losses.BinaryCrossentropy(from_logits=False)})
print(model.summary())
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/")

# Run eagerly
#model.run_eagerly = True
model.fit(train_dataset.batch(32), epochs=50, validation_data=test_dataset.batch(32), callbacks=[early_stop, tensorboard])
# Save
model.save("blurry_to_real_model2.keras")

# Visualize reconstructions from the test dataset
random_indices = np.random.randint(0, 200, 5)
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
    err = reconstruct_error(shape, reconstr_shape)
    ax[1].set_title("Purely algorithmic reconstruction using X degrees of measures")
    ax[1].set_xlabel(f"Reconstruction error: {err :.2f}")
    
    # Plot the model's reconstruction
    model_reconstr = model.predict(reconstr_shape.reshape((1,*reconstr_shape.shape)))[0]
    print(f"Model reconstr shape: {model_reconstr.shape}")
    print(f"shape shape: {shape.shape}")
    model_reconstr = model_reconstr.reshape(shape.shape)
    rec_error = reconstruct_error(shape, model_reconstr)
    ax[2].imshow(model_reconstr)
    ax[2].set_title("U-net reconstruction from the simple reconstruction")
    ax[2].set_xlabel(f"Reconstruction error: {rec_error :.2f}")
plt.show()
    
    

