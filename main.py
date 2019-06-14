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
from medpy.metric.binary import dc, hd
import keras
import keras.preprocessing as kp
import tensorflow as tf

data_augm = False

whichloss = 'binary_crossentropy'
# whichdatasets = ['York', 'ACDC']
whichdataset = 'ACDC'


whichmodels = ['twolayernetwork']
# whichmodels = ['param_unet']



seeds = [1, 2, 3, 4] # for reproducibility
# seeds = [1]
batch_size = 32
data_percs = [0.25, 0.5, 0.75, 1]  # between 0 and 1, not percentages
# data_percs = [0.25]
slice_percs = [0.25, 0.5, 0.75, 1]
filters = 64
splits = {1: (0.3, 0.1)}  # values for test and validation percentages

epochs = 100
threshold = 0.5


amount_training_data_split1 = []
amount_training_data_split2 = []
amount_training_data_split3 = []
amount_training_data_split4 = []



if whichdataset == 'York':
    input = np.load('YCMRI_128x128_images.npy', allow_pickle=True)
    labels = np.load('YCMRI_128x128_labels.npy', allow_pickle=True)
    path = 'York_results'
    # plt.figure()
    # for i in range(len(labels)):
    #     plt.hist(np.unique(labels[i][:]))
    #
    # plt.show()


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


    input = np.load('unet_input.npy', allow_pickle=True)
    labels = np.load('unet_labels.npy', allow_pickle=True)

    # plt.figure()
    # for i in range(len(labels)):                      #verfy that input labels are 0 and 1
    #     plt.hist(np.unique(labels[i][:]))
    #
    # plt.show()
    old_input = input
    old_labels = labels

data_dict = []




unet_input = []
unet_labels = []



for split in splits:
    for whichmodel in whichmodels:
        if whichmodel == 'param_unet':
            layers_arr = [4 ,5 ,6, 7]
            # layers_arr = [2]

        else:
            layers_arr = [1]
        for layers in layers_arr:
            if layers==6:
                batch_size = batch_size//2
            if layers > 6:
                batch_size = batch_size//4
            for perc_index, perc in enumerate(data_percs):
                temp_slice_perc_split1 = []
                temp_slice_perc_split2 = []
                temp_slice_perc_split3 = []
                temp_slice_perc_split4 = []

                for slice_perc in slice_percs:
                    for split_number in seeds:
                        random.seed(split_number)
                        np.random.seed(split_number)
                        tf.set_random_seed(split_number)
                        os.environ['PYTHONHASHSEED'] = str(split_number)



                        train_pats, test_pats, val_pats = methods.get_patient_split(len(input), splits.get(split))

                        train_pats = methods.get_total_perc_pats(train_pats, perc)
                        test_pats = methods.get_total_perc_pats(test_pats, 1)
                        val_pats = methods.get_total_perc_pats(val_pats, 1)

                        x_train, y_train= methods.get_patient_perc_split(input, labels, train_pats, slice_perc)
                        x_test, y_test= methods.get_patient_perc_split(input, labels, test_pats, 1, test = True)
                        x_val, y_val= methods.get_patient_perc_split(input, labels, val_pats, 1)



                        if split_number == 1:
                            if slice_perc == 0.25:
                                temp_slice_perc_split1.append(x_train.shape[0])
                            if slice_perc == 0.5:
                                temp_slice_perc_split1.append(x_train.shape[0])
                            if slice_perc == 0.75:
                                temp_slice_perc_split1.append(x_train.shape[0])
                            if slice_perc == 1:
                                temp_slice_perc_split1.append(x_train.shape[0])

                        if split_number == 2:
                            if slice_perc == 0.25:
                                temp_slice_perc_split2.append(x_train.shape[0])
                            if slice_perc == 0.5:
                                temp_slice_perc_split2.append(x_train.shape[0])
                            if slice_perc == 0.75:
                                temp_slice_perc_split2.append(x_train.shape[0])
                            if slice_perc == 1:
                                temp_slice_perc_split2.append(x_train.shape[0])

                        if split_number == 3:
                            if slice_perc == 0.25:
                                temp_slice_perc_split3.append(x_train.shape[0])
                            if slice_perc == 0.5:
                                temp_slice_perc_split3.append(x_train.shape[0])
                            if slice_perc == 0.75:
                                temp_slice_perc_split3.append(x_train.shape[0])
                            if slice_perc == 1:
                                temp_slice_perc_split3.append(x_train.shape[0])

                        if split_number == 4:
                            if slice_perc == 0.25:
                                temp_slice_perc_split4.append(x_train.shape[0])
                            if slice_perc == 0.5:
                                temp_slice_perc_split4.append(x_train.shape[0])
                            if slice_perc == 0.75:
                                temp_slice_perc_split4.append(x_train.shape[0])
                            if slice_perc == 1:
                                temp_slice_perc_split4.append(x_train.shape[0])

                amount_training_data_split1.append(temp_slice_perc_split1)
                amount_training_data_split2.append(temp_slice_perc_split2)
                amount_training_data_split3.append(temp_slice_perc_split3)
                amount_training_data_split4.append(temp_slice_perc_split4)

amount_training_data_York = [amount_training_data_split1, amount_training_data_split2, amount_training_data_split3, amount_training_data_split4]
"""
amount_training_data: array where x: splits, y: person_percentage, z: slice_percentage
for example : amount_training_data_ACDC[0][0][0] is for split 1, 0,25% for person and slice percentage
"""
something = 0

np.save('amount_training_data_ACDC', amount_training_data_York)