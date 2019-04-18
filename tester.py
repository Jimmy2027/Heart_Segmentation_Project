import network_trainer as nt
import acdc_data_loader as acdc
import methods
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn import model_selection

unet_input = np.load('unet_input.npy')
unet_labels = np.load('unet_labels.npy')

x_train, x_test , y_train, y_test = model_selection.train_test_split(unet_input,unet_labels, test_size= 0.3)

