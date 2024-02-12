""" With this file, we train a model to take in a blurry image (reconstructed from measures)
and output the real image.
"""
import tensorflow as tf
import os
from reconstruct import rollback_reconstruct_circle
import numpy as np
import matplotlib.pyplot as plt
import keras
from utils import reconstruct_error
from tensorflow_models import create_a_U_net_model
import tqdm

# GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

folder = "Circle_radius_40_scans"
# There are files named "circle_<n>.npy" and "measurement_<n>.npy" in the folder
# Lets find all of them
measurement_paths = [os.path.join(folder, f"measurements_{i}.npy") for i in range(5000)]
circle_paths = [os.path.join(folder, f"circle_{i}.npy") for i in range(5000)]

# Each measurement contains 360 measurements (one for each angle), and each circle is 64x64
# To train the model, we take measures from  -90 to 90 degrees, and use those to create the first reconstruction
# The model then takes in the reconstruction and outputs the real image (circle_paths)

# Create a dataset from the paths
def make_dataset_from_first(measurement_paths, circle_paths):
    def generator():
        for measurement_path, circle_path,i in zip(measurement_paths, circle_paths, range(len(measurement_paths))):
            try:
                measurements = np.load(measurement_path)
            except:
                print(f"Could not load {measurement_path}")
                continue
            # Randomly choose a 30 - 180 degree slice
            slice_length = np.random.randint(30, 180)
            start_index = np.random.randint(0, 359 - slice_length)
            end_index = start_index + slice_length
            # Get the measurements
            measurements = measurements[start_index:end_index]
            # Reconstruct the circle
            reconstr_circle = rollback_reconstruct_circle(measurements, np.arange(start_index, end_index), 40).matrix
            # Get the real circle
            circle = np.load(circle_path)
            yield reconstr_circle, circle, i
        return
    ds = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.int32))
    return ds

if False:
    dataset = make_dataset_from_first(measurement_paths, circle_paths)
    # Loop through the dataset and save the data
    for i, (reconstr_circle, circle, ith) in tqdm.tqdm(enumerate(dataset)):
        np.save(os.path.join(folder, f"reconstructed_circle_{ith}.npy"), reconstr_circle)

# Load the new dataset
reconstructed_circle_paths = [os.path.join(folder, f"reconstructed_circle_{i}.npy") for i in range(4600)]
circle_paths = [os.path.join(folder, f"circle_{i}.npy") for i in range(4600)]

def make_dataset_from_second(reconstructed_circle_paths, circle_paths):
    def generator():
        for reconstructed_circle_path, circle_path in zip(reconstructed_circle_paths, circle_paths):
            try:
                reconstructed_circle = np.load(reconstructed_circle_path)
                circle = np.load(circle_path)
            except:
                #print(f"Could not load {reconstructed_circle_path} or {circle_path}")
                continue
            yield reconstructed_circle, circle
        return
    ds = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32))
    return ds



dataset = make_dataset_from_second(reconstructed_circle_paths, circle_paths)

# Print sizes
first_sample = list(dataset.take(1))[0]
input_shape = first_sample[0].shape
print(f"First sample shape: {first_sample[0].shape}")
print(f"Second sample shape: {first_sample[1].shape}")

train_dataset = dataset.take(4000)
test_dataset = dataset.skip(4000)


#model = create_a_U_net_model(input_shape=input_shape)
model = tf.keras.models.load_model("model_r40.keras")

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
    reconstr_circle, circle = list(test_dataset.skip(i).take(1))[0]
    reconstr_circle = reconstr_circle.numpy()
    circle = circle.numpy()
    # Plot the reconstruction
    fig, ax = plt.subplots(1,3)
    # Plot the real circle
    ax[0].imshow(circle)
    ax[0].set_title("Ground truth object")
    
    ax[1].imshow(reconstr_circle)
    err = reconstruct_error(circle, reconstr_circle)
    ax[1].set_title("Purely algorithmic reconstruction using 130 degrees of measures")
    ax[1].set_xlabel(f"Reconstruction error: {err :.2f}")
    
    # Plot the model's reconstruction
    model_reconstr = model.predict(reconstr_circle.reshape((1,81,81)))[0]
    print(f"Model reconstr shape: {model_reconstr.shape}")
    print(f"circle shape: {circle.shape}")
    model_reconstr = model_reconstr.reshape(circle.shape)
    rec_error = reconstruct_error(circle, model_reconstr)
    ax[2].imshow(model_reconstr)
    ax[2].set_title("U-net reconstruction from the simple reconstruction")
    ax[2].set_xlabel(f"Reconstruction error: {rec_error :.2f}")
plt.show()
    
    

