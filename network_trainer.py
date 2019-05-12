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
from sklearn.metrics import roc_curve, auc

whichdataset = 'ACDC'
# whichdataset = 'York'
# whichmodel = 'param_unet'
whichmodel = 'twolayernetwork'

number_of_patients = 10
filters = 64
# max 5 layers with 96x96
layers = 2




if whichdataset == 'York':
    unet_input = np.load('YCMRI_128x128_images.npy')
    unet_labels = np.load('YCMRI_128x128_labels.npy')
    unet_input = np.concatenate(unet_input, axis = 0)
    unet_labels = np.concatenate(unet_labels, axis= 0)
    unet_input = np.expand_dims(unet_input, -1)
    unet_labels = np.expand_dims(unet_labels, -1)
    path = 'York_results'

if whichdataset == 'ACDC':

    data_dir = 'ACDC_dataset/training'
    raw_image_path01 = '_frame01.nii.gz'
    raw_image_path12 = '_frame12.nii.gz'
    label_path01 = '_frame01_gt.nii.gz'
    label_path12 = '_frame12_gt.nii.gz'
    path = 'ACDC_results'


images_paths, labels_paths = methods.load_images(data_dir, raw_image_path01, raw_image_path12, label_path01, label_path12)
images_paths.sort()   # each label should be on the same index than its corresponding image
labels_paths.sort()

img_data = methods.load_data(images_paths)

input = np.load('unet_input.npy')
labels = np.load('unet_labels.npy')

#TODO variable amount of slices per person? (in percentages)
total_number_of_patients = len(input)
unet_input = []
unet_labels = []
random_patient_num = [random.sample(range(total_number_of_patients-1), number_of_patients)]

for i in random_patient_num[0]:
    unet_input.append(input[i])
    unet_labels.append(labels[i])

x_train, x_test , y_train, y_test = model_selection.train_test_split(unet_input, unet_labels, test_size=0.3)


x_train = np.concatenate(x_train, axis = 0)
y_train = np.concatenate(y_train, axis = 0)

x_train = np.expand_dims(x_train, -1)  #-1 for the last element
y_train = np.expand_dims(y_train, -1)



resolution = x_train.shape[1]

print(np.shape(x_train), np.shape(y_train))

# conv 2D default parameter: channels last: (batch, rows, cols, channels)

validation_split_val = 0.25
batch_size = 32
epochs = 1
input_size = (resolution, resolution, 1)
kernel_size = 3
Dropout_rate = 0.5





if whichmodel == 'param_unet':
    model = unet.param_unet(input_size, filters, layers)


if whichmodel == 'twolayernetwork':
    model = unet.twolayernetwork(input_size, kernel_size, Dropout_rate)





if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(path + '/' + whichmodel):
    os.makedirs(path + '/' + whichmodel)
if not os.path.exists(path + '/'+whichmodel+'/'+str(number_of_patients)+'patients'):
    os.makedirs(path + '/'+whichmodel+'/'+ str(number_of_patients)+'patients')
if not os.path.exists(path + '/'+whichmodel+'/'+ str(number_of_patients)+'patients/' + str(layers)+'layers'):
    os.makedirs(path + '/'+whichmodel+'/'+ str(number_of_patients)+'patients/' + str(layers)+'layers')
save_dir = path + '/' + whichmodel +'/' + str(number_of_patients)+'patients/'+ str(layers)+'layers'


if whichmodel == 'param_unet' or whichmodel == 'unet':

    model_checkpoint = ModelCheckpoint(save_dir+'/unet.{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True)
    history = model.fit(x_train, y_train, steps_per_epoch=1, epochs=8, callbacks=[model_checkpoint])
else:
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_split=validation_split_val, epochs=epochs, verbose=1)













print(history.history.keys())


#
# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_accuracy_values.png'))
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_loss_values.png'))
#
# model.save(os.path.join(save_dir, str(epochs) +'epochs_test.h5'))

y_pred = []
for i in x_test:
    i = np.expand_dims(i, -1)
    y_pred.append(model.predict(i, verbose = 1))

np.save(os.path.join(save_dir, str(epochs) + 'epochs_y_pred'), y_pred)



thresholds = [0.1, 0.25, 0.5, 0.75]

for threshold in thresholds:

    upper, lower = 1, 0

    results = {
        "median_ROC_AUC": "median_ROC_AUC",
        "dice": "dice",
        "roc_auc": "roc_auc",
        "median_dice_score": "median_dice_score"
    }
    dice = []
    roc_auc = []
    output = []

    for i in range(len(y_test)):
        output.append(np.squeeze(y_pred[i]))
        for s in range(y_test[i].shape[0]):
            output[i][s] = np.where(output[i][s] > threshold, upper, lower)

            dice.append(su.dice(output[i][s], y_test[i][s]))
            y_true = y_test[i][s].reshape(-1)
            y_pred_temp = y_pred[i][s].reshape(-1)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_temp)
            roc_auc.append(auc(fpr, tpr))


    median_ROC_AUC = np.median(roc_auc)
    median_dice_score = np.median(dice)

    results["median_dice_score"] = median_dice_score
    results["median_ROC_AUC"] = median_ROC_AUC
    results["dice"] = dice
    results["roc_auc"] = roc_auc
    torch.save(results, os.path.join(save_dir, str(epochs) + 'epochs_evaluation_results'))

    print('DICE SCORE: ' + str(median_dice_score))
    print('ROC AUC:', str(median_ROC_AUC))

    methods.save_datavisualisation3(x_test, y_test, output, os.path.join(save_dir,str(threshold) + 'thr_'), True, True)





