import network_trainer as nt
import acdc_data_loader as acdc
import methods
import numpy as np
from keras.models import load_model
















model = load_model('unet.hdf5')
results = model.predict(x_test, verbose = 1)

np.save('results', results)