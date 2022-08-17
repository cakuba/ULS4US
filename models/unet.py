"""
    Implementation of vanilla UNet and MIMO-UNet in Python 3.6.9/Keras 2.4.0/Tensorflow 2.4.1 
"""

# to filter some unnecessory warning messages
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import (Input, concatenate, Conv2D, MaxPooling2D, AveragePooling2D,
                         Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization, Lambda)

from tensorflow.python.keras import backend_config
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops

epsilon = backend_config.epsilon

def _constant_to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.

  This is slightly faster than the _to_tensor function, at the cost of
  handling fewer cases.

  Arguments:
      x: An object to be converted (numpy arrays, floats, ints and lists of
        them).
      dtype: The destination type.

  Returns:
      A tensor.
  """
  return constant_op.constant(x, dtype=dtype)

def Accuracy(y_true, y_pred, thres=0.5):
    """
    to compute the customized metric as Accuracy
    """

    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    thres = math_ops.cast(thres, y_pred.dtype)
    y_pred = math_ops.cast(y_pred > thres, y_pred.dtype)
    return K.mean(math_ops.equal(y_true, y_pred), axis=-1)

def Dice(y_true, y_pred):
    """
    to compute the customized metric as Dice Coefficient
    """
    return real_dice_coef(y_true, y_pred)

def real_dice_coef(y_true, y_pred):
    """
    to compute the Dice Coef. between the human annoation and network prediction 
    
    Argumetns:
       y_true - human annotion in the format of [batch_size, width, height]
       y_pred - network prediction in the format of [batch_size, width, height] 
    
    """

    smooth = 1.0e-5
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    intersection = math_ops.reduce_sum(math_ops.multiply(y_true, y_pred),axis=[1,2])
    y_true_sum = math_ops.reduce_sum(math_ops.multiply(y_true, y_true),axis=[1,2])
    y_pred_sum = math_ops.reduce_sum(math_ops.multiply(y_pred, y_pred),axis=[1,2])
    dice = (2.0*intersection+smooth)/(y_true_sum+y_pred_sum+smooth)

    return dice

def binary_crossentropy(y_true, y_pred):
    """
    Computes the binary crossentropy loss.
    """
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return math_ops.reduce_mean(K.binary_crossentropy(y_true, y_pred), axis=[1,2])
    #return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def weighted_binary_crossentropy(y_true, y_pred, beta):
    """
    Computes the weighted binary crossentropy loss with beta=[0,1]

    Note:
        beta>0.5, 减少假阴性
        beta<0.5，减少假阳性
    """
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    beta_ = _constant_to_tensor(beta, y_pred.dtype.base_dtype)
    epsilon_ = _constant_to_tensor(epsilon(), y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

    # Compute cross entropy from probabilities.
    bce = beta_ * y_true * math_ops.log(y_pred + epsilon())
    bce += (1-beta_) * (1 - y_true) * math_ops.log(1 - y_pred + epsilon())

    return math_ops.reduce_mean(-bce, axis=[1,2])

def UNet_vanilla(img_width=256,img_height=256,dropout=0.2,seed=2020,show_summary=False):
    # to randomly initialize the weights
    kernel_initializer=initializers.he_normal(seed=seed)

    inputs = Input((img_width, img_height, 1),name='input')
    # -------------------------------------------------------------------------------------------------------------
    conv1_0 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1_1 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1_0)
    bn1 = BatchNormalization()(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2))(bn1)
    # -------------------------------------------------------------------------------------------------------------
    conv2_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2_0)
    bn2 = BatchNormalization()(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2))(bn2)
    # -------------------------------------------------------------------------------------------------------------
    conv3_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3_0)
    bn3 = BatchNormalization()(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2))(bn3)
    # -------------------------------------------------------------------------------------------------------------
    conv4_0 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4_0)
    bn4 = BatchNormalization()(conv4_1)
    drop4 = Dropout(dropout)(bn4)
    # end of down-sampling
    # -------------------------------------------------------------------------------------------------------------
    # start of up-sampling

    up5 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([conv3_1, up5], axis=3)
    conv5_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge5)
    conv5_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5_0)
    bn5 = BatchNormalization()(conv5_1)
    # -------------------------------------------------------------------------------------------------------------
    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(bn5))
    merge6 = concatenate([conv2_1, up6], axis=3)
    conv6_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6_0)
    bn6 = BatchNormalization()(conv6_1)
    # -------------------------------------------------------------------------------------------------------------
    up7 = Conv2D(64 , 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(bn6))
    merge7 = concatenate([conv1_1, up7], axis=3)
    conv7_0 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7_1 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7_0)
    bn7 = BatchNormalization()(conv7_1)
    conv7 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(bn7)
    # ----------------------------------------------------------------------------------------------------------
    output = Conv2D(1, 1, activation='sigmoid', name='output')(conv7)

    model = Model(inputs=inputs, outputs=output)

    if show_summary:
        model.summary()

    return model

def mimo_loss(target1, target2, output1, output2, output1_cropped, mode=1, beta1=0.5, beta2=0.5):
    """
    Compute the customized loss for mimo networks

    mode=0, binaryCrossEntropy loss
    mode=1, weighted binaryCrossEntropy
    mode=2, weighted binaryCrossEntropy + self-supervised binaryCrossEntroy
    """

    if (mode==0):
        loss_bce1 = binary_crossentropy(target1, output1)
        loss_bce2 = binary_crossentropy(target2, output2)
        return loss_bce1+loss_bce2
    elif (mode==1):
        loss_weighted_bce1 = weighted_binary_crossentropy(target1,output1,beta=beta1)
        loss_weighted_bce2 = weighted_binary_crossentropy(target2,output2,beta=beta2)
        return loss_weighted_bce1+loss_weighted_bce2
    elif (mode==2):
        loss_weighted_bce1 = weighted_binary_crossentropy(target1,output1,beta=beta1)
        loss_weighted_bce2 = weighted_binary_crossentropy(target2,output2,beta=beta2)

        loss_interOutputs = binary_crossentropy(output1_cropped, output2)
        return loss_weighted_bce1+loss_weighted_bce2+loss_interOutputs

def UNet_mimo(img1_width=512,img1_height=512,
                         img2_width=256,img2_height=256,
                         dropout=0.2,seed=2022,loss_mode=1,beta1=0.5,beta2=0.5,show_summary=False):
    """
        Implementation of MIMO-UNet with customized loss functions
    """
    # to randomly initialize the weights
    kernel_initializer=initializers.he_normal(seed=seed)

    # Encoder1 for Original Image
    input1 = Input((img1_width, img1_height, 1),name='input1')
    input1_target = Input((img1_width, img1_height, 1),name='input1_target')
    # -------------------------------------------------------------------------------------------------------------
    conv1_0 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(input1)
    conv1_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1_0)
    bn1_1 = BatchNormalization()(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2))(bn1_1)
    # -------------------------------------------------------------------------------------------------------------
    conv2_0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2_0)
    bn1_2 = BatchNormalization()(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2))(bn1_2)
    # -------------------------------------------------------------------------------------------------------------
    conv3_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3_0)
    bn1_3 = BatchNormalization()(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2))(bn1_3)
    # -------------------------------------------------------------------------------------------------------------
    conv4_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4_0)
    bn1_4 = BatchNormalization()(conv4_1)
    pool4 = MaxPooling2D(pool_size=(2))(bn1_4)
    # -------------------------------------------------------------------------------------------------------------
    conv5_0 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
    conv5_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5_0)
    bn1_5 = BatchNormalization()(conv5_1)
    drop1 = Dropout(dropout)(bn1_5)
    # end of down-sampling for the Encoder1

    # Encoder2 for Cropped Image
    input2 = Input((img2_width, img2_height, 1),name='input2')
    input2_target = Input((img2_width, img2_height, 1),name='input2_target')
    # -------------------------------------------------------------------------------------------------------------
    conv1_0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(input2)
    conv1_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1_0)
    bn2_1 = BatchNormalization()(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2))(bn2_1)
    # -------------------------------------------------------------------------------------------------------------
    conv2_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2_0)
    bn2_2 = BatchNormalization()(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2))(bn2_2)
    # -------------------------------------------------------------------------------------------------------------
    conv3_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3_0)
    bn2_3 = BatchNormalization()(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2))(bn2_3)
    # -------------------------------------------------------------------------------------------------------------
    conv4_0 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4_0)
    bn2_4 = BatchNormalization()(conv4_1)
    drop2 = Dropout(dropout)(bn2_4)
    # end of down-sampling for the Encoder2

    # -------------------------------------------------------------------------------------------------------------
    # Concatenate of outputs from Encoder1 and Encoder2
    merge1 = concatenate([drop1, drop2], axis=3)
    # -------------------------------------------------------------------------------------------------------------

    # Decoder for merged features
    # start of up-sampling
    up1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(merge1))
    merge2 = concatenate([bn1_4, bn2_3, up1], axis=3)
    convd1_0 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge2)
    convd1_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(convd1_0)
    bnd1 = BatchNormalization()(convd1_1)    

    up2 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(bnd1))
    merge3 = concatenate([bn1_3, bn2_2, up2], axis=3)
    convd2_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge3)
    convd2_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(convd2_0)
    bnd2 = BatchNormalization()(convd2_1)

    up3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(bnd2))
    merge4 = concatenate([bn1_2, bn2_1, up3], axis=3)
    convd3_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge4)
    convd3_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(convd3_0)    
    bnd3 = BatchNormalization()(convd3_1)

    up4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(bnd3))
    merge4 = concatenate([bn1_1, up4], axis=3)
    convd4_0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge4)
    convd4_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(convd4_0)  
    bnd4 = BatchNormalization()(convd4_1)
    # end of up-sampling
    # ----------------------------------------------------------------------------------------------------------

    convd5 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(bnd4)
    output1 = Conv2D(1, 1, activation='sigmoid', name='output1')(convd5)

    convd6 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(bnd3)
    output2 = Conv2D(1, 1, activation='sigmoid', name='output2')(convd6)
    # end of outputs

    # -------------------------------------------------------------------------------------------------------------
    # the definition of loss layer
    output1_cropped = Input((img2_width, img2_height, 1),name='output1_cropped')
    customized_loss = Lambda(lambda x: mimo_loss(*x,mode=loss_mode,beta1=beta1,beta2=beta2),name="mimo_loss")([input1_target,input2_target,output1,output2,output1_cropped])

    # finalize model inputs/outputs
    model = Model(inputs=[input1,input1_target,input2,input2_target,output1_cropped], outputs=[output1,output2,customized_loss])

    # to compute the customized loss
    layer = model.get_layer("mimo_loss")
    loss = tf.reduce_sum(layer.output, keepdims=True)
    model.add_loss(loss)

    # to compute the customized metrics
    model.add_metric(Accuracy(input1_target,output1), name="Orig Acc")
    model.add_metric(Accuracy(input2_target,output2), name="Crop Acc")
    model.add_metric(Dice(input1_target,output1), name="Orig Dice")
    model.add_metric(Dice(input2_target,output2), name="Crop Dice")

    if show_summary:
        model.summary()

    return model