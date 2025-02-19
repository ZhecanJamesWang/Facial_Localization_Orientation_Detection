from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

def model(weights_path=None):
    model = Sequential()
    model.add(Convolution2D(16, 4, 4, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(32, 4, 4, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(48, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(234, activation='relu'))
    model.add(Dense(7))

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    # im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    # im[:,:,0] -= 103.939
    # im[:,:,1] -= 116.779
    # im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1))
    # im = np.expand_dims(im, axis=0)

    # Test pretrained model
    # model = VGG_16('vgg16_weights.h5')
    model = VGG_16()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print np.argmax(out)

