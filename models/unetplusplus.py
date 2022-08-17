"""
    implementation of UNet++ in Python 3.6.9/Keras 2.4.0/Tensorflow 2.4.1 
    Ref: https://github.com/MrGiovanni/UNetPlusPlus
"""

from tensorflow.keras.layers import Conv2D,Activation,Conv2DTranspose,UpSampling2D,BatchNormalization,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, VGG19
import numpy as np
import copy

DEFAULT_SKIP_CONNECTIONS = {
    'vgg16':            ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2',
                         'block5_pool', 'block4_pool', 'block3_pool', 'block2_pool', 'block1_pool',
                        ),
    'vgg19':            ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2',
                         'block5_pool', 'block4_pool', 'block3_pool', 'block2_pool', 'block1_pool',
                        ),
    'resnet18':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                         'relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                        ),
    'resnet34':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                         'relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                        ),
    'resnet50':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                         'relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                        ),
    'resnet101':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                         'relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                        ),
    'resnet152':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                         'relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                        ),
    'resnext50':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                         'stage4_unit1_relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                        ),
    'resnext101':       ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                         'stage4_unit1_relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                        ),
    'inceptionv3':          (228, 86, 16, 9),
    'inceptionresnetv2':    (594, 260, 16, 9),
    'densenet121':          (311, 139, 51, 4),
    'densenet169':          (367, 139, 51, 4),
    'densenet201':          (479, 139, 51, 4),
}

backbones = {
    "vgg16": VGG16,
    "vgg19": VGG19,
}

def get_layer_number(model, layer_name):
    """
    Help find layer in Keras model by name
    Args:
        model: Keras `Model`
        layer_name: str, name of layer
    Returns:
        index of layer
    Raises:
        ValueError: if model does not contains layer with such name
    """
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))

def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))

def handle_block_names(stage, cols):
    conv_name = 'decoder_stage{}-{}_conv'.format(stage, cols)
    bn_name = 'decoder_stage{}-{}_bn'.format(stage, cols)
    relu_name = 'decoder_stage{}-{}_relu'.format(stage, cols)
    up_name = 'decoder_stage{}-{}_upsample'.format(stage, cols)
    merge_name = 'merge_{}-{}'.format(stage, cols)
    return conv_name, bn_name, relu_name, up_name, merge_name


def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x
    return layer


def Upsample2D_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            if type(skip) is list:
                x = Concatenate(name=merge_name)([x] + skip)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer


def Transpose2D_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            # print("\nskip = {}".format(skip))
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)
                for l in skip:
                    merge_list.append(l)
                x = Concatenate(name=merge_name)(merge_list)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return 

def build_xnet(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True):

    input = backbone.input
    # print(n_upsample_blocks)

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    if len(skip_connection_layers) > n_upsample_blocks:
        downsampling_layers = skip_connection_layers[int(len(skip_connection_layers)/2):]
        skip_connection_layers = skip_connection_layers[:int(len(skip_connection_layers)/2)]
    else:
        downsampling_layers = skip_connection_layers

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])
    skip_layers_list = [backbone.layers[skip_connection_idx[i]].output for i in range(len(skip_connection_idx))]
    downsampling_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in downsampling_layers])
    downsampling_list = [backbone.layers[downsampling_idx[i]].output for i in range(len(downsampling_idx))]
    downterm = [None] * (n_upsample_blocks+1)
    for i in range(len(downsampling_idx)):
        if downsampling_list[0].shape == backbone.output.shape:
            downterm[n_upsample_blocks-i] = downsampling_list[i]
        else:
            downterm[n_upsample_blocks-i-1] = downsampling_list[i]
    downterm[-1] = backbone.output

    interm = [None] * (n_upsample_blocks+1) * (n_upsample_blocks+1)
    for i in range(len(skip_connection_idx)):
        interm[-i*(n_upsample_blocks+1)+(n_upsample_blocks+1)*(n_upsample_blocks-1)] = skip_layers_list[i]
    interm[(n_upsample_blocks+1)*n_upsample_blocks] = backbone.output

    for j in range(n_upsample_blocks):
        for i in range(n_upsample_blocks-j):
            upsample_rate = to_tuple(upsample_rates[i])
            # print(j, i)
            
            if i == 0 and j < n_upsample_blocks-1 and len(skip_connection_layers) < n_upsample_blocks:
                interm[(n_upsample_blocks+1)*i+j+1] = None
            elif j == 0:
                if downterm[i+1] is not None:
                    interm[(n_upsample_blocks+1)*i+j+1] = up_block(decoder_filters[n_upsample_blocks-i-2], 
                                      i+1, j+1, upsample_rate=upsample_rate,
                                      skip=interm[(n_upsample_blocks+1)*i+j], 
                                      use_batchnorm=use_batchnorm)(downterm[i+1])
                else:
                    interm[(n_upsample_blocks+1)*i+j+1] = None
            else:
                interm[(n_upsample_blocks+1)*i+j+1] = up_block(decoder_filters[n_upsample_blocks-i-2], 
                                  i+1, j+1, upsample_rate=upsample_rate,
                                  skip=interm[(n_upsample_blocks+1)*i : (n_upsample_blocks+1)*i+j+1], 
                                  use_batchnorm=use_batchnorm)(interm[(n_upsample_blocks+1)*(i+1)+j])
                
    """
    for i in range(n_upsample_blocks-2):
        interm = []
        x = skip_layers_list[n_upsample_blocks-i-2]
    
    x = {}
    for stage in range(n_upsample_blocks-1):
        i = n_upsample_blocks - stage - 1
        x = backbone.layers[skip_connection_idx[i-1]].output
        for col in range(stage+1):
            print("i = {}, col = {}, index = {}".format(i, col, i+col))
            skip_connection = None
            if i-col < len(skip_connection_idx):
                skip_connection = skip_layers_list[i-col]
            upsample_rate = to_tuple(upsample_rates[i-col])
            x = up_block(decoder_filters[i-col], stage-col+1, col+1, upsample_rate=upsample_rate,
                         skip=skip_connection, use_batchnorm=use_batchnorm)(x)
            skip_layers_list[i+col] = x
    x = backbone.output
    for i in range(n_upsample_blocks):
        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            # skip_connection = backbone.layers[skip_connection_idx[i]].output
            skip_connection = skip_layers_list[i]
        upsample_rate = to_tuple(upsample_rates[i])
        x = up_block(decoder_filters[i], n_upsample_blocks-i, 0, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)
    """

    """
    i = n_upsample_blocks - 1
    xx = backbone.layers[skip_connection_idx[i-0-1]].output
    skip_connection = skip_layers_list[i-0]
    upsample_rate = to_tuple(upsample_rates[i-0])
    xx = up_block(decoder_filters[i-0], n_upsample_blocks-i-0, 1+0, upsample_rate=upsample_rate,
                 skip=skip_connection, use_batchnorm=use_batchnorm)(xx)
    skip_layers_list[i-0] = xx
    i = n_upsample_blocks - 2
    xx = backbone.layers[skip_connection_idx[i-0-1]].output
    skip_connection = skip_layers_list[i-0]
    upsample_rate = to_tuple(upsample_rates[i-0])
    xx = up_block(decoder_filters[i-0], n_upsample_blocks-i-0, 1+0, upsample_rate=upsample_rate,
                 skip=skip_connection, use_batchnorm=use_batchnorm)(xx)
    skip_layers_list[i-0] = xx
    skip_connection = skip_layers_list[i-1]
    upsample_rate = to_tuple(upsample_rates[i-1])
    xx = up_block(decoder_filters[i-1], n_upsample_blocks-i-1, 1+1, upsample_rate=upsample_rate,
                 skip=skip_connection, use_batchnorm=use_batchnorm)(xx)
    skip_layers_list[i-1] = xx
    """
    
    """
    for i in range(n_upsample_blocks):
        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            # skip_connection = backbone.layers[skip_connection_idx[i]].output
            skip_connection = skip_layers_list[i]
        upsample_rate = to_tuple(upsample_rates[i])
        x = up_block(decoder_filters[i], n_upsample_blocks-i, 0, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)
    """

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(interm[n_upsample_blocks])
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model

def Xnet(backbone_name='vgg16',
         input_shape=(None, None, 3),
         input_tensor=None,
         encoder_weights='imagenet',
         skip_connections='default',
         decoder_block_type='upsampling',
         decoder_filters=(256,128,64,32,16),
         decoder_use_batchnorm=True,
         n_upsample_blocks=5,
         upsample_rates=(2,2,2,2,2),
         classes=1,
         activation='sigmoid'):
    """
    Args:
        backbone_name: (str) look at list of available backbones.
        input_shape:  (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization), 
            'imagenet' (pre-training on ImageNet), 
            'dof' (pre-training on DoF)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        skip_connections: if 'default' is used take default skip connections,
            else provide a list of layer numbers or names starting from top of model
        decoder_block_type: (str) one of 'upsampling' and 'transpose' (look at blocks.py)
        decoder_filters: (int) number of convolution layer filters in decoder blocks
        decoder_use_batchnorm: (bool) if True add batch normalisation layer between `Conv2D` ad `Activation` layers
        n_upsample_blocks: (int) a number of upsampling blocks
        upsample_rates: (tuple of int) upsampling rates decoder blocks
        classes: (int) a number of classes for output
        activation: (str) one of keras activations for last model layer
    Returns:
        keras.models.Model instance
    """

    backbone = backbones[backbone_name](
                        input_shape=input_shape,
                        input_tensor=input_tensor,
                        weights=encoder_weights,
                        include_top=False)

    if skip_connections == 'default':
        skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name]
    # n_upsample_blocks = len(skip_connections)

    model = build_xnet(backbone,
                       classes,
                       skip_connections,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=n_upsample_blocks,
                       upsample_rates=upsample_rates,
                       use_batchnorm=decoder_use_batchnorm)

    model._name = 'xnet-{}'.format(backbone_name)
    return model