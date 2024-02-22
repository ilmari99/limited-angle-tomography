""" With this file, we train a model to take in a blurry image (reconstructed from measures)
and output the real image.
"""
import tensorflow as tf
import os
from AbsorptionMatrices import Square
from reconstruct import rollback_reconstruct_shape
import numpy as np
import matplotlib.pyplot as plt
import keras
from utils import reconstruct_error
from tensorflow_models import create_a_U_net_model
import tqdm
from create_dataset import create_dataset

# GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


folder = "DatascansTest2"

#create_dataset(folder, 64, 20, (1,3), portion_missing_pixels_bounds=(0.05,0.5), hole_size_volatility=0.1)

# There are files named "shape_<n>.npy" and "measurement_<n>.npy" in the folder
# Lets find all of them
measurement_paths = [os.path.join(folder, f"measurements_{i}.npy") for i in range(20)]
shape_paths = [os.path.join(folder, f"shape_{i}.npy") for i in range(20)]

# Each measurement contains 360 measurements (one for each angle), and each shape is 64x64
# To train the model, we take measures from  -90 to 90 degrees, and use those to create the first reconstruction
# The model then takes in the reconstruction and outputs the real image (shape_paths)

# Create a dataset from the paths
def make_dataset_from_first(measurement_paths, shape_paths):
    def generator():
        for measurement_path, shape_path,i in zip(measurement_paths, shape_paths, range(len(measurement_paths))):
            try:
                measurements = np.load(measurement_path)
                shape = np.load(shape_path)
            except:
                print(f"Could not load {measurement_path} or {shape_path}")
                continue
            # Randomly choose a 30 - 180 degree slice
            slice_length = np.random.randint(170, 220)
            start_index = np.random.randint(0, 359 - slice_length)
            end_index = start_index + slice_length
            # Get the measurements
            measurements = measurements[start_index:end_index]
            base_shape = Square(measurements.shape[1])
            print(f"Reconstructing shape {i} from {start_index} to {end_index} degrees")
            print(f"Shape: {shape.shape}, measurements: {measurements.shape}")
            print(f"base_shape: {base_shape.matrix.shape}")
            # Reconstruct the shape
            reconstr_shape = rollback_reconstruct_shape(measurements, np.arange(start_index, end_index), base_shape).matrix
            yield reconstr_shape, shape, i, slice_length
        return
    ds = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.int32, tf.int32))
    return ds

# Firstly, we load the measurements, and original shapes.
# We the reconstruct the shapes from the measurements, and save them to the folder
# We only do this if necessary
if True:
    dataset = make_dataset_from_first(measurement_paths, shape_paths)
    # Save the reconstructions
    for i, (reconstr_shape, shape, ith, nangles) in tqdm.tqdm(enumerate(dataset)):
        np.save(os.path.join(folder, f"reconstructed_shape_{ith}_a{nangles}.npy"), reconstr_shape)

# Load the new dataset
# Find all paths with "reconstructed_shape" int the name
reconstructed_shape_paths_unsorted = list(filter(lambda x: "reconstructed_shape" in x, os.listdir(folder)))
# Sort the paths by the number in the name
is_ = [int(x.split("_")[2]) for x in reconstructed_shape_paths_unsorted]
reconstructed_shape_paths = sorted(reconstructed_shape_paths_unsorted, key=lambda x: int(x.split("_")[2]))
# Load the shape paths
shape_paths = [os.path.join(folder, f"shape_{i}.npy") for i in is_]
exit()

def make_dataset_from_second(reconstructed_shape_paths, shape_paths):
    def generator():
        for reconstructed_shape_path, shape_path in zip(reconstructed_shape_paths, shape_paths):
            try:
                reconstructed_shape = np.load(reconstructed_shape_path)
                shape = np.load(shape_path)
            except:
                #print(f"Could not load {reconstructed_shape_path} or {shape_path}")
                continue
            yield reconstructed_shape, shape
        return
    ds = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32))
    return ds



dataset = make_dataset_from_second(reconstructed_shape_paths, shape_paths)

# Print sizes
first_sample = list(dataset.take(1))[0]
input_shape = first_sample[0].shape
print(f"First sample shape: {first_sample[0].shape}")
print(f"Second sample shape: {first_sample[1].shape}")

train_dataset = dataset.take(4000)
test_dataset = dataset.skip(4000)


#model = create_a_U_net_model(input_shape=input_shape)
model = tf.keras.models.load_model("model_r40.keras")
print(model.summary())
exit()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/")

# Run eagerly
#model.run_eagerly = True
#model.fit(train_dataset.batch(64), epochs=20, validation_data=test_dataset.batch(64), callbacks=[early_stop, tensorboard])
# Save
#model.save(f"model_r40.keras")

# Visualize reconstructions from the test dataset
random_indices = np.random.randint(0, 40, 10)
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
    ax[1].set_title("Purely algorithmic reconstruction using 130 degrees of measures")
    ax[1].set_xlabel(f"Reconstruction error: {err :.2f}")
    
    # Plot the model's reconstruction
    model_reconstr = model.predict(reconstr_shape.reshape((1,81,81)))[0]
    print(f"Model reconstr shape: {model_reconstr.shape}")
    print(f"shape shape: {shape.shape}")
    model_reconstr = model_reconstr.reshape(shape.shape)
    rec_error = reconstruct_error(shape, model_reconstr)
    ax[2].imshow(model_reconstr)
    ax[2].set_title("U-net reconstruction from the simple reconstruction")
    ax[2].set_xlabel(f"Reconstruction error: {rec_error :.2f}")
plt.show()
    
    

