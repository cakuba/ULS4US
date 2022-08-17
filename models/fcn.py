"""
    Implementation of FCN in Python 3.6.9/Keras 2.4.0/Tensorflow 2.4.1 
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def vgg16(l2=0, dropout=0):
    '''Convolutionized VGG16 network. 
    Args:
      l2 (float): L2 regularization strength
      dropout (float): Dropout rate
    Returns:
      (keras Model)
    '''
    ## Input
    input_layer = keras.Input(shape=(None, None, 3), name='input')
    ## Preprocessing
    x = keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input, name='preprocessing')(input_layer)
    ## Block 1
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block1_conv1')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3,  strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block1_conv2')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block1_pool')(x)
    ## Block 2
    x = keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block2_conv1')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3,  strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block2_conv2')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block2_pool')(x)
    ## Block 3
    x = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block3_conv1')(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block3_conv2')(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block3_conv3')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block3_pool')(x)
    ## Block 4
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block4_conv1')(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block4_conv2')(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block4_conv3')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block4_pool')(x)
    ## Block 5
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block5_conv1')(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block5_conv2')(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='block5_conv3')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block5_pool')(x)
    ## Convolutionized fully-connected layers
    x = keras.layers.Conv2D(filters=4096, kernel_size=(7,7), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='conv6')(x)
    x = keras.layers.Dropout(rate=dropout, name='drop6')(x)
    x = keras.layers.Conv2D(filters=4096, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.L2(l2=l2), name='conv7')(x)
    x = keras.layers.Dropout(rate=dropout, name='drop7')(x)
    ## Inference layer
    x = keras.layers.Conv2D(filters=1000, kernel_size=(1,1), strides=(1,1), padding='same', activation='sigmoid', 
                            name='pred')(x)
    return keras.Model(input_layer, x)

def fcn32(vgg16, l2=0):
    '''32x upsampled FCN.
    Args:
      vgg16 (keras Model): VGG16 model to build upon
      l2 (float): L2 regularization strength
    Returns:
      (keras Model)
    '''
    x = keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='same', activation='linear',
                            kernel_regularizer=keras.regularizers.L2(l2=l2),
                            name='score7')(vgg16.get_layer('drop7').output)
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=(64,64), strides=(32,32),
                                     padding='same', use_bias=False, activation='sigmoid',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=keras.regularizers.L2(l2=l2),
                                     name='fcn32')(x)
    return keras.Model(vgg16.input, x)

def fcn16(vgg16, fcn32, l2=0):
    '''16x upsampled FCN.
    Args:
      vgg16 (keras Model): VGG16 model to build upon
      fcn32 (keras Model): FCN32 model to build upon
      l2 (float): L2 regularization strength
    Returns:
      (keras Model)
    '''
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=(4,4), strides=(2,2),
                                     padding='same', use_bias=False, activation='linear',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=keras.regularizers.L2(l2=l2),
                                     name='score7_upsample')(fcn32.get_layer('score7').output)
    y = keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='same', activation='linear',
                            kernel_initializer=keras.initializers.Zeros(),
                            kernel_regularizer=keras.regularizers.L2(l2=l2),
                            name='score4')(vgg16.get_layer('block4_pool').output)
    x = keras.layers.Add(name='skip4')([x, y])
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=(32,32), strides=(16, 16),
                                     padding='same', use_bias=False, activation='sigmoid',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=keras.regularizers.L2(l2=l2),
                                     name='fcn16')(x)
    return keras.Model(fcn32.input, x)

def fcn8(vgg16, fcn16, l2=0):
    '''8x upsampled FCN.
    Args:
      vgg16 (keras Model): VGG16 model to build upon
      fcn16 (keras Model): FCN16 model to build upon
      l2 (float): L2 regularization strength
    Returns:
      (keras Model)
    '''
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=(4,4), strides=(2,2),
                                     padding='same', use_bias=False, activation='linear',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=keras.regularizers.L2(l2=l2),
                                     name='skip4_upsample')(fcn16.get_layer('skip4').output)
    y = keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='same', activation='linear',
                            kernel_initializer=keras.initializers.Zeros(),
                            kernel_regularizer=keras.regularizers.L2(l2=l2),
                            name='score3')(vgg16.get_layer('block3_pool').output)
    x = keras.layers.Add(name='skip3')([x, y])
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=(16,16), strides=(8,8),
                                     padding='same', use_bias=False, activation='sigmoid',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=keras.regularizers.L2(l2=l2),
                                     name='fcn8')(x)
    return keras.Model(fcn16.input, x)



## ================
## Misc functions for training
## ================

class BilinearInitializer(keras.initializers.Initializer):
    '''Initializer for Conv2DTranspose to perform bilinear interpolation on each channel.'''
    def __call__(self, shape, dtype=None, **kwargs):
        kernel_size, _, filters, _ = shape
        arr = np.zeros((kernel_size, kernel_size, filters, filters))
        ## make filter that performs bilinear interpolation through Conv2DTranspose
        upscale_factor = (kernel_size+1)//2
        if kernel_size % 2 == 1:
            center = upscale_factor - 1
        else:
            center = upscale_factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        kernel = (1-np.abs(og[0]-center)/upscale_factor) * \
                 (1-np.abs(og[1]-center)/upscale_factor) # kernel shape is (kernel_size, kernel_size)
        for i in range(filters):
            arr[..., i, i] = kernel
        return tf.convert_to_tensor(arr, dtype=dtype)