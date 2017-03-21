# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers.core import Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

from keras.utils.layer_utils import convert_all_kernels_in_model
# from keras.regularizers.WeightRegularizer import *
from keras.regularizers import l2, activity_l2


def model(input_shape = None, weights_path = None):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # if weights not in {'imagenet', None}:
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization) or `imagenet` '
    #                      '(pre-training on ImageNet).')

    # if weights == 'imagenet' and include_top and classes != 1000:
    #     raise ValueError('If using `weights` as imagenet with `include_top`'
    #                      ' as true, `classes` should be 1000')
    # # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=48,
    #                                   data_format=K.image_data_format(),
    #                                   include_top=include_top)

    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    # Block 1
    img_input = Input(shape=input_shape)    
    x = Convolution2D(16, 4, 4, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    # x = Convolution2D(16, 4, 4, activation='relu', border_mode='same', name='block1_conv2')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(32, 4, 4, activation='relu', border_mode='same', name='block2_conv1')(x)
    # x = Convolution2D(32, 4, 4, activation='relu', border_mode='same', name='block2_conv2')(x)    
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(48, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    # x = Convolution2D(48, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)    
    # x = Convolution2D(48, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)    
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    # x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    # x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)  
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    # x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    # x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)  
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5pool')(x)

    x = Flatten(name='flatten')(x)
    # x = Dense(256, activation='relu', name='fc1')(x)
    # x = Dropout(0.2,name='fc1_drop')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.2,name='fc1_drop')(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.2,name='fc2_drop')(x)
    x = Dense(3, activation = 'linear', name='predLabel')(x)

    model = Model(img_input, x, name='customizedModel')
    
    if weights_path:
       model.load_weights(weights_path, by_name = True)

    return model


if __name__ == '__main__':
    pass
    # model = VGG16(include_top=True, weights='imagenet')

    # img_path = 'elephant.jpg'
    # img = image.load_img(img_path, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # print('Input image shape:', x.shape)

    # preds = model.predict(x)
    # print('Predicted:', decode_predictions(preds))
