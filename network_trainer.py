import Unet as unet
from matplotlib import pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint

unet_input = np.load('unet_input.npy')
unet_labels = np.load('unet_labels.npy')

print(np.shape(unet_input), np.shape(unet_labels))

def network_trainer(cropper_size):
    """
    cropped_img_data and cropped_myocar_labels: (64,64)
    """
    validation_split_val = 0.25
    batch_size = 32
    epochs = 1
    input_size = (2*cropper_size, 2*cropper_size, 1)

    model = unet.unet(input_size)
    model.summary()
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit(unet_input, unet_labels, validation_split=validation_split_val, validation_steps = 10, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    # history = model.fit(unet_input, unet_labels, validation_split=validation_split_val, batch_size=batch_size, nb_epoch=epochs, verbose=1)
    #
    # print(history.history.keys())

    model.save('test.h5')

