import tensorflow as tf
import os
import keras

""" Code adapted from https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
"""

def _double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def _downsample_block(x, n_filters):
   f = _double_conv_block(x, n_filters)
   p = keras.layers.MaxPool2D(2)(f)
   p = keras.layers.Dropout(0.3)(p)
   return f, p

def _upsample_block(x, conv_features, n_filters):
    # upsample
    x = keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    
    # Hacky fix to make sure the shapes match.
    # I'm not sure why this happens
    if x.shape != conv_features.shape:
        # Skip the first value, except on batch size and channels
        diff0 = conv_features.shape[1] - x.shape[1]
        diff1 = conv_features.shape[2] - x.shape[2]
        conv_features = conv_features[:,diff0:,diff1:,:]
    
    # concatenate
    x = keras.layers.concatenate([x, conv_features])
    # dropout
    x = keras.layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = _double_conv_block(x, n_filters)
    return x

def create_a_U_net_model(input_shape = (129,129)):
    """ Create a U-Net architecture.
    The input is a flawed image, and the output is the corrected image.
    """
    # Input layer
    inputs = keras.layers.Input(shape=input_shape)
    # Expand the dimensions
    x = tf.reshape(inputs, (-1,input_shape[0],input_shape[1],1))
    # Downsample
    f1, p1 = _downsample_block(x, 16)
    f2, p2 = _downsample_block(p1, 32)
    f3, p3 = _downsample_block(p2, 64)
    f4, p4 = _downsample_block(p3, 128)
    
    # Bottleneck
    b = _double_conv_block(p4, 256)
    
    # Upsample
    u1 = _upsample_block(b, f4, 128)
    u2 = _upsample_block(u1, f3, 64)
    u3 = _upsample_block(u2, f2, 32)
    u4 = _upsample_block(u3, f1, 16)
    
    # Output layer
    outputs = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u4)
    # Pad with one zero on the 2nd and 3rd dimensions
    #outputs = tf.pad(outputs, [[0,0],[1,1],[1,1],[0,0]])
    # Reshape the output
    outputs = tf.image.resize(outputs, input_shape)
    #outputs = tf.reshape(outputs, (-1,input_shape[0],input_shape[1]))
    #pad to input_shape
    #outputs = tf.pad(outputs, [[0,0], [1,1], [2,1]])
    # Create the model
    model = keras.models.Model(inputs = inputs, outputs = outputs)
    # Compile the model
    loss = "binary_crossentropy"
    model.compile(optimizer = "adam", loss = loss, metrics = ["accuracy"])
    return model