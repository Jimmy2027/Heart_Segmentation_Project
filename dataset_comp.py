import models as unet
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
from medpy.metric.binary import dc, hd
import keras
import keras.preprocessing as kp
import tensorflow as tf
from email_notification import NotificationSystem

which_train_dataset = 'ACDC'
which_test_dataset = 'York'
whichmodel = 'param_unet'
whichloss = 'binary_crossentropy'
layers = 4
splits = {1: (0.3, 0.1)}
seeds = [1, 2, 3]
pers_percs = [0.25, 0.5, 0.75, 1]
slice_perc = 1

if which_test_dataset == 'York':
    input = np.load('YCMRI_128x128_images.npy', allow_pickle=True)
    labels = np.load('YCMRI_128x128_labels.npy', allow_pickle=True)




if which_test_dataset== 'ACDC':

    data_dir = 'ACDC_dataset/training'
    raw_image_path01 = '_frame01.nii.gz'
    raw_image_path12 = '_frame12.nii.gz'
    label_path01 = '_frame01_gt.nii.gz'
    label_path12 = '_frame12_gt.nii.gz'

    images_paths, labels_paths = methods.load_images(data_dir, raw_image_path01, raw_image_path12, label_path01,
                                                     label_path12)
    images_paths.sort()  # each label should be on the same index than its corresponding image
    labels_paths.sort()

    img_data = methods.load_data(images_paths)

    input = np.load('unet_input.npy', allow_pickle=True)
    labels = np.load('unet_labels.npy', allow_pickle=True)
    path = 'ACDC_results'

data_dict = []

unet_input = []
unet_labels = []

path = which_train_dataset + '_predicts_' + which_test_dataset+ '_results'

for perc_index, perc in enumerate(pers_percs):
    for split in splits:
        for split_number in seeds:
            random.seed(split_number)
            np.random.seed(split_number)
            tf.set_random_seed(split_number)
            os.environ['PYTHONHASHSEED'] = str(split_number)

            train_pats, test_pats, val_pats = methods.get_patient_split(len(input), splits.get(split))



            x_test, y_test = methods.get_patient_perc_split(input, labels, test_pats, 1, test=True)

            if not os.path.exists(
                    path + '/' + whichmodel + '/' + whichloss + '/' + str(pers_percs[perc_index]) + 'patients/' + '/' + str(
                            slice_perc) + 'slices' + '/' + str(layers) + 'layers/' + str(split_number) + 'split'):
                os.makedirs(path + '/' + whichmodel + '/' + whichloss + '/' + str(pers_percs[perc_index]) + 'patients/' + '/' + str(
                    slice_perc) + 'slices' + '/' + str(layers) + 'layers/' + str(split_number) + 'split')

            save_dir = path + '/' + whichmodel + '/' + whichloss + '/' + str(pers_percs[perc_index]) + 'patients/' + '/' + str(
                slice_perc) + 'slices' + '/' + str(layers) + 'layers/' + str(split_number) + 'split'


            for file in os.listdir(os.path.join(which_train_dataset+ '_results', 'param_unet', 'binary_crossentropy','1patients', '1slices', '4layers', str(seeds[0]) + 'split')):
                if file.endswith('hdf5'):
                    model_path = os.path.join(which_train_dataset+ '_results', 'param_unet','binary_crossentropy', '1patients', '1slices', '4layers', str(seeds[0]) + 'split', file)


            model = keras.models.load_model(model_path)

            y_pred = []
            for i in x_test:
                i = np.expand_dims(i, -1)
                y_pred.append(model.predict(i, verbose=1))

            threshold = 0.5

            dice = []
            dice_thresholded = []
            roc_auc = []
            nonthr_hausdorff = []
            thresholded_hausdorff = []
            thresholded_y_pred = []
            output = []

            thresholded_output = []

            np.save(os.path.join(save_dir, 'y_pred'), y_pred)
            np.save(os.path.join(save_dir, 'y_test'), y_test)

            for i in range(len(y_test)):
                thresholded_y_pred.append(np.where(y_pred[i] > threshold, 1, 0))
            for i in range(len(y_test)):
                thresholded_output.append(np.squeeze(thresholded_y_pred[i]))
                output.append(np.squeeze(y_pred[i]))
                for s in range(y_test[i].shape[0]):
                    dice.append(dc(output[i][s], y_test[i][s]))
                    dice_thresholded.append(dc(thresholded_output[i][s], y_test[i][s]))
                    if np.max(output[i][s]) != 0:
                        nonthr_hausdorff.append(hd(output[i][s], y_test[i][s]))
                    if np.max(thresholded_output[i][s]) != 0:
                        thresholded_hausdorff.append(hd(thresholded_output[i][s], y_test[i][s])) #causes an error if thresholded is 0
                    y_true = y_test[i][s].reshape(-1)
                    y_pred_temp = thresholded_y_pred[i][s].reshape(-1)

            median_dice = np.median(dice_thresholded)
            median_hausdorff = np.mean(thresholded_hausdorff)

            methods.save_datavisualisation3(x_test, y_test, output, thresholded_output,str(round(median_dice, 4)), save_dir+'/', True, True)



something = 0
