import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def segnetwork(img_shape, kernel_size, Dropout_rate):
    model = Sequential()

    # Encoder Layers
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    # Decoder Layers
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(2,1, activation='relu', padding='same'))  #try this
    model.add(Conv2D(1, 1, activation='sigmoid', padding='same'))

    whichmodel = 'segnet'

    return model, whichmodel

def param_unet(input_size, filters, layers, pretrained_weights=None):
    inputs = Input(input_size)
    print('Inputs in UNet Shape: ' + str(inputs.shape))
    conv_down = np.empty(layers, dtype=object)
    conv_up = np.empty(layers, dtype=object)
    temp = inputs
    for i in range(layers):
        print('----------Layer #' + str(i+1) + '----------')
        print('First Conv with ' + str(filters * 2**i) + ' filters at index ' + str(i))
        conv_down[i] = Conv2D(filters * 2**i, 3, activation='relu', padding='same', kernel_initializer='he_normal')(temp)
        print('Second Conv with ' + str(filters * 2**i) + ' filters at index ' + str(i))
        conv_down[i] = Conv2D(filters * 2**i, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_down[i])
        if i < layers-1:
            print('MaxPooling')
            temp = MaxPooling2D(pool_size=(2, 2))(conv_down[i])
        print('End Layer #' + str(i+1))

    for j in range(layers-2, -1, -1):
        print('----------Layer #' + str(j+1) + '----------')
        print('First Conv with ' + str(filters * 2**j) + ' filters at index ' + str(j))
        temp = Conv2D(filters * 2 ** j, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv_down[j+1]))
        print('Merging ' + str(conv_down[j].shape) + ' and ' + str(temp.shape))
        merge6 = concatenate([conv_down[j], temp], axis=3)
        print('Second Conv with ' + str(filters * 2**j) + ' filters at index ' + str(j))
        conv_up[j] = Conv2D(filters * 2 ** j, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            merge6)
        print('Third Conv with ' + str(filters * 2 ** j) + ' filters at index ' + str(j))
        conv_up[j] = Conv2D(filters * 2 ** j, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            conv_up[j])
        print('End Layer #' + str(j + 1))

    conv_almost_final = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_up[0])
    conv_final = Conv2D(1, 1, activation='sigmoid')(conv_almost_final)
    print('********** Resulting shape: ' + str(conv_final.shape) + ' **********')

    model = Model(input=inputs, output=conv_final)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# TODO: try dice loss

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet(input_size, pretrained_weights=None):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    #try dice
    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    whichmodel = 'unet'
    return model, whichmodel

#
def smallsegnetwork(img_shape, kernel_size, Dropout_rate):
    model = Sequential()

    # Encoder Layers
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))

    # Decoder Layers
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(1, 1, activation='sigmoid', padding='same'))

    whichmodel = 'smallsegnetwork'
    return model, whichmodel


def twolayernetwork(img_shape, kernel_size, Dropout_rate):
    model = Sequential()

    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(1, kernel_size, activation='sigmoid', padding='same'))
    model.add(BatchNormalization())


    return model


if __name__ == '__main__':      #only gets called if Unet.py is run

    model = unet((128,128,1), 64, 9)

    from keras.utils import plot_model

    plot_model(model, to_file='param_unet.png', show_shapes=True)
