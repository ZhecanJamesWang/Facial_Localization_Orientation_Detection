from keras.layers import Input
from keras.applications import VGG19
from keras import *
import cv2

import numpy as np


# -*- coding: utf-8 -*-
'''VGG19 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

# '''
# from __future__ import print_function
# from __future__ import absolute_import

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D

from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import optimizers
from random import shuffle
import os
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import * #decode_predictions, preprocess_input, _obtain_input_shape
from keras.regularizers.WeightRegularizer import *
from keras.regularizers import l2, activity_l2
# from keras.applications.imagenet_utils import

TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


os.environ["CUDA_VISIBLE_DEVICES"]="1"

def VGG19Var(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None):
    '''Instantiate the VGG19 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
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
            has to be `(224, 224, 3)` (with `tf` dim ordering)
            or `(3, 224, 244)` (with `th` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.

    # Returns
        A Keras model instance.
    '''
    # if weights not in {'imagenet', None}:
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization) or `imagenet` '
    #                      '(pre-training on ImageNet).')
    # Determine proper input shape
    input_shape = (224,224,3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    DR=0.001
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1',W_regularizer=l2(DR))(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2',W_regularizer=l2(DR))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1',W_regularizer=l2(DR))(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2',W_regularizer=l2(DR))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1',W_regularizer=l2(DR))(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2',W_regularizer=l2(DR))(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3',W_regularizer=l2(DR))(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv4',W_regularizer=l2(DR))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1',W_regularizer=l2(DR))(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2',W_regularizer=l2(DR))(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3',W_regularizer=l2(DR))(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv4',W_regularizer=l2(DR))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1',W_regularizer=l2(DR))(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2',W_regularizer=l2(DR))(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3',W_regularizer=l2(DR))(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv4',W_regularizer=l2(DR))(x)
    x = MaxPooling2D((5,5), strides=(3, 3), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(2048, activation='relu', name='fc1_ST')(x)
        x = Dense(1024, activation='relu', name='fc2_ST')(x)
        # x = Dense(1000, activation='softmax', name='predictions')(x)
        x = Dense(136, activation='linear',name='predS')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg19')

    # load weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('vgg19_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path,by_name=True)

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path,by_name=True)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    elif weights!=None:
        print "use speicific model path"
        if K.image_dim_ordering() == 'th':
           print "Th not considered here. TensorFlow not set up properly"
        else:
            if include_top:
                weights_path = weights
            else:
                weights_path = weights
            model.load_weights(weights_path, by_name=True)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model




def DataGen(DataStrs, BatchSize,imSize=224,isTrain=True,L3DIdx=136):
    InputData = np.zeros([BatchSize,imSize,imSize,3],dtype=np.float32)
    InputLabel = np.zeros([BatchSize,136],dtype=np.float32)

    InputNames=[]
    MaxIters = len(DataStrs)/BatchSize
    for Mi in range(MaxIters):
        train_start = BatchSize*Mi
        train_end = train_start+BatchSize
        count = 0
        for i in range(train_start,train_end):
            strLine = DataStrs[i]
            strCells = strLine.rstrip(' \n').split(' ')
            imgName = strCells[0]
            labels = np.array(strCells[1:]).astype(np.float)
            if isTrain:
                labels=labels[:L3DIdx]
            im = cv2.resize(cv2.imread(imgName), (imSize, imSize)).astype(np.float32)
            im[:, :, 0] -= 103.939
            im[:, :, 1] -= 116.779
            im[:, :, 2] -= 123.68
            InputData[count,...]=im.copy()
            InputLabel[count,...]=labels
            InputNames.append(imgName)
            count=count+1
        yield InputData,InputLabel

def load_train_data(DataStrs, train_start, train_end, n_training_examples,imSize=224,isTrain=True,L3DIdx=136):

    InputData = np.zeros([n_training_examples,imSize,imSize,3],dtype=np.float32)
    InputLabel = np.zeros([n_training_examples,136],dtype=np.float32)
    count=0
    InputNames=[]
    for i in range(train_start,train_end):
        strLine = DataStrs[i]
        strCells = strLine.rstrip(' \n').split(' ')
        imgName = strCells[0]
        labels = np.array(strCells[1:]).astype(np.float)
        if isTrain:
            labels=labels[:L3DIdx]
        im = cv2.resize(cv2.imread(imgName), (imSize, imSize)).astype(np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        InputData[count,...]=im.copy()
        InputLabel[count,...]=labels
        InputNames.append(imgName)
        count=count+1

    return InputData,InputLabel,InputNames

    #
    # X_train = HDF5Matrix(datapath, 'features', train_start, train_start+n_training_examples, normalizer=normalize_data)
    # y_train = HDF5Matrix(datapath, 'targets', train_start, train_start+n_training_examples)

    # return X_train, y_train, X_test, y_test
def PrednSave(fileName,P2d,SavePath):
    cvImg = cv2.imread(fileName)
    # cvImg = cv2.cvtColor(cvImg1, cv2.COLOR_BGR2RGB)
    a=np.arange(0,68)
    a[60]=48
    a[64]=54
    b=np.arange(1,69)
    b[17-1]=16-1
    b[22-1]=21-1
    b[27-1]=27-1
    b[42-1]=37-1
    b[48-1]=43-1
    b[36-1]=31-1
    b[60-1]=49-1
    b[68-1]=48#61-1

    for k in range(68):
        P1=(int(P2d[a[k], 0]), int(P2d[a[k], 1]))
        P2= (int(P2d[b[k], 0]), int(P2d[b[k], 1]))
        cv2.circle(cvImg, P1, 3, (0,255,0), -1)
        cv2.line(cvImg, P1,P2,(0,0,255),2,-1)
    cv2.imwrite(SavePath,cvImg)

# model = VGG19Var(weights='imagenet')
model = VGG19Var(weights='./temp.h5')
# im = cv2.resize(cv2.imread('cat.jpeg'), (224, 224)).astype(np.float32)
# im[:,:,0] -= 103.939
# im[:,:,1] -= 116.779
# im[:,:,2] -= 123.68
# # im = im.transpose((2,0,1))
# im = np.expand_dims(im, axis=0)
# val=model.predict(im)
# print val.shape
model.summary()
# from matplotlib.pyplot import *
# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png')



sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=sgd)

nb_epoch=2

TrainPath = '/home/shengtao/Data/2D_Images/Croped256/Script/KBKC4_train.txt'
TestPath = '/home/shengtao/Data/2D_Images/300W/300WP5CropTest.txt'
FTr = open(TrainPath,'r')
DataTr = FTr.readlines()
FTe = open(TestPath,'r')
DataTe = FTe.readlines()
# TrData,TrLabel=load_train_data(DataTr,0,5,5)

batch_size=8
TrNum = len(DataTr)
MaxIters = TrNum/batch_size


for e in range(nb_epoch):
    # if e>0:
    shuffle(DataTr)
    for iter in range (MaxIters):
        train_start=iter*batch_size
        train_end = (iter+1)*batch_size
        X_batch, Y_batch, Z_Names = load_train_data(DataTr,train_start,train_end,batch_size)
        loss = model.train_on_batch(X_batch, Y_batch)
        if iter%50==0:
            print np.float(iter/MaxIters)*100,iter,loss
            # model.save('./temp.h5')
            TeX, TeY,imgNames = load_train_data(DataTe, iter%len(DataTe), iter%len(DataTe)+1, 1)
            pos=model.predict(TeX).reshape([68,2])*256+128
            PrednSave(imgNames[0],pos,'./temp.jpg')
        if iter%2000==0 and iter !=0:

            model.save('./temp.h5')
            # print TeY-pos
# # Alternatively, without data augmentation / normalization:
# for e in range(nb_epoch):
#     print("epoch %d" % e)
#     for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
#         model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)


#model.fit_generator() ###