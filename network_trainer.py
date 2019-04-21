import Unet as unet
from matplotlib import pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
import acdc_data_loader as acdc
from sklearn import model_selection


resolution = acdc.get_resolution()

unet_input = np.load('unet_input.npy')
unet_labels = np.load('unet_labels.npy')

x_train, x_test , y_train, y_test = model_selection.train_test_split(unet_input,unet_labels, test_size= 0.3)

np.save('x_test', x_test)
np.save('y_test', y_test)

#temp something

print(np.shape(x_train), np.shape(y_train))

# conv 2D default parameter: channels last: (batch, rows, cols, channels)

validation_split_val = 0.25
batch_size = 32
epochs = 10
input_size = (resolution, resolution, 1)

model = unet.unet(input_size)
model.summary()
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, validation_split=validation_split_val, epochs=epochs, callbacks=[model_checkpoint], verbose =1)
print(history.history.keys())



model.save('test.h5')


y_pred = model.predict(x_test)

np.save('y_pred', y_pred)

plt.figure(figsize=(resolution, resolution))

for i in range(0, 6):
    # plot original image
    ax = plt.subplot(2, 6, i + 1)
    plt.imshow(x_test[i,:,:,0], plt.cm.gray)

    # plot reconstructed image
    ax = plt.subplot(2, 6, 6 + i + 1)
    plt.imshow(y_pred[i,:,:,0], plt.cm.gray)

plt.savefig('results.png', bbox_inches='tight')




