import network_trainer as nt
import acdc_data_loader as acdc
import methods
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import acdc_data_loader as acdc


x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

resolution = acdc.get_resolution()

model = load_model('unet.hdf5')
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

