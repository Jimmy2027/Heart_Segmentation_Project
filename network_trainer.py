import Unet as unet
from matplotlib import pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn import model_selection
import scoring_utils as su
import torch
import os
import json
import pandas as pd
import methods
import random

# unet_input = np.load('YCMRI_128x128_images.npy')
# unet_labels = np.load('YCMRI_128x128_labels.npy')
# unet_input = np.concatenate(unet_input, axis = 0)
# unet_labels = np.concatenate(unet_labels, axis= 0)
# unet_input = np.expand_dims(unet_input, -1)
# unet_labels = np.expand_dims(unet_labels, -1)

data_dir = 'ACDC_dataset/training'
raw_image_path01 = '_frame01.nii.gz'
raw_image_path12 = '_frame12.nii.gz'
label_path01 = '_frame01_gt.nii.gz'
label_path12 = '_frame12_gt.nii.gz'


images_paths, labels_paths = methods.load_images(data_dir, raw_image_path01, raw_image_path12, label_path01, label_path12)
images_paths.sort()   # each label should be on the same index than its corresponding image
labels_paths.sort()

img_data = methods.load_data(images_paths)

input = np.load('unet_input.npy')
labels = np.load('unet_labels.npy')

number_of_patients = 100 #TODO variable amount of slices per person? (in percentages)
total_number_of_patients = len(input)
unet_input = []
unet_labels = []
for i in range(0, number_of_patients):
    print(i)
    rand = random.randint(0, total_number_of_patients-1)
    unet_input.append(input[rand])
    unet_labels.append(labels[rand])


x_train, x_test , y_train, y_test = model_selection.train_test_split(unet_input, unet_labels, test_size=0.3)


# num_train_persons = int(0.7 * number_of_patients)
# num_test_persons = number_of_patients - num_train_persons
#
# # ind = np.random.permutation(np.shape(unet_input)[0])
# # unet_input = unet_input[ind]
# # print(unet_input.shape)
# # unet_labels = unet_labels[ind]
# # print(unet_labels.shape)
#
# x_train = []
# x_test = []
# y_train = []
# y_test = []
#
# for i in range(0, num_train_persons):
#     x_train.append(unet_input[i])
#     y_train.append(unet_labels[i])
#
# for i in range(num_train_persons, number_of_patients):
#     x_test.append(unet_input[i])
#     y_test.append(unet_labels[i])


x_train = np.concatenate(x_train, axis = 0)
y_train = np.concatenate(y_train, axis = 0)

x_train = np.expand_dims(x_train, -1)  #-1 for the last element
y_train = np.expand_dims(y_train, -1)



resolution = x_train.shape[1]



np.save('x_test', x_test)
np.save('y_test', y_test)


print(np.shape(x_train), np.shape(y_train))

# conv 2D default parameter: channels last: (batch, rows, cols, channels)

validation_split_val = 0.25
batch_size = 32
epochs = 10
input_size = (resolution, resolution, 1)
kernel_size = 3
Dropout_rate = 0.5




# model, whichmodel = unet.unet(input_size)
model, whichmodel = unet.segnetwork(input_size, kernel_size, Dropout_rate)


if not os.path.exists('Results/' + whichmodel):
    os.makedirs('Results/' + whichmodel)
if not os.path.exists('Results/'+whichmodel+'/'+str(number_of_patients)+'patients'):
    os.makedirs('Results/'+whichmodel+'/'+ str(number_of_patients)+'patients')
save_dir = 'Results/' + whichmodel +'/' + str(number_of_patients)+'patients'

# model_checkpoint = ModelCheckpoint(os.path.join(save_dir, str(epochs) +'epochs_unet.hdf5'), monitor='loss', verbose=1, save_best_only=True)
# history = model.fit(x_train, y_train, validation_split=validation_split_val, epochs=epochs, callbacks=[model_checkpoint], verbose =1)






model.summary()
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=validation_split_val, epochs=epochs, verbose = 1)







print(history.history.keys())



# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_accuracy_values.png'))

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_loss_values.png'))

model.save(os.path.join(save_dir, str(epochs) +'epochs_test.h5'))

y_pred = []
for i in x_test:
    i = np.expand_dims(i, -1)
    y_pred.append(model.predict(i))

np.save(os.path.join(save_dir,str(epochs) + 'epochs_x_test'), x_test)



# torch.save(results, os.path.join(save_dir,str(epochs) +'epochs_evaluation_results'))



threshold, upper, lower = 0.5, 1, 0
dice = []



for i in range(np.shape(y_test)[0]):
    y_pred[i] = np.squeeze(y_pred[i])
    for s in range(y_test[i].shape[0]):
        y_pred[i][s] = np.where(y_pred[i][s] > threshold, upper, lower)

        dice.append(su.dice(y_pred[i][s], y_test[i][s]))



dice_score = np.median(dice)

print('DICE SCORE: ' + str(dice_score))


np.save(os.path.join(save_dir,str(epochs) +'epochs_y_pred'), y_pred)
methods.save_datavisualisation3(x_test, y_test, y_pred, os.path.join(save_dir, str(number_of_patients)+'pat' + str(epochs) + 'epochs_'), True, True)





