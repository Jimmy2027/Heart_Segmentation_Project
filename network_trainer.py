import Unet as unet
from matplotlib import pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
import acdc_data_loader as acdc
from sklearn import model_selection


cropper_size = acdc.get_cropper_size()

unet_input = np.load('unet_input.npy')
unet_labels = np.load('unet_labels.npy')

x_train, x_test , y_train, y_test = model_selection.train_test_split(unet_input,unet_labels, test_size= 0.3)

print(np.shape(x_train), np.shape(y_train))

# conv 2D default parameter: channels last: (batch, rows, cols, channels)

validation_split_val = 0.25
batch_size = 32
epochs = 1
input_size = (2*cropper_size, 2*cropper_size, 1)

model = unet.unet(input_size)
model.summary()
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, validation_split=validation_split_val, epochs=epochs, callbacks=[model_checkpoint], verbose =1)
print(history.history.keys())



model.save('test.h5')

