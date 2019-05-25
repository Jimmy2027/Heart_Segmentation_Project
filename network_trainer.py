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
import keras

# whichloss = 'binary_crossentropy'
whichloss = 'dice'
whichdataset = 'ACDC'
# whichdataset = 'York'
whichmodel = 'param_unet'
# whichmodel = 'unet'

# whichmodel = 'twolayernetwork'
# whichmodel = 'segnetwork'


# number_of_patients = 10
filters = 64
# max 5 layers with 96x96
layers_arr = [8,7,6,5,4,3,2]

# layers_arr = [1]
epochs = 100


all_results = []




if whichdataset == 'York':
    input = np.load('YCMRI_128x128_images.npy')
    labels = np.load('YCMRI_128x128_labels.npy')
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


    input = np.load('unet_input.npy', allow_pickle=True)
    labels = np.load('unet_labels.npy', allow_pickle=True)


unet_input = []
unet_labels = []
#TODO variable amount of slices per person? (in percentages)

total_number_of_patients = len(input)
# total_number_of_patients = 20

arr_number_of_patients = [total_number_of_patients - 1]

for layers in layers_arr:
    for number_of_patients in arr_number_of_patients:

        random_patient_num = [random.sample(range(total_number_of_patients-1), number_of_patients)]

        for i in random_patient_num[0]:
            unet_input.append(input[i])
            unet_labels.append(labels[i])

        x_train, x_test , y_train, y_test = model_selection.train_test_split(unet_input, unet_labels, test_size=0.3)



        x_train = np.concatenate(x_train, axis = 0)
        y_train = np.concatenate(y_train, axis =0)

        x_train = np.expand_dims(x_train, -1)  #-1 for the last element
        y_train = np.expand_dims(y_train, -1)



        resolution = x_train.shape[1]

        print(np.shape(x_train), np.shape(y_train))

        # conv 2D default parameter: channels last: (batch, rows, cols, channels)

        validation_split_val = 0.25
        batch_size = 32
        input_size = (resolution, resolution, 1)
        kernel_size = 3
        Dropout_rate = 0.5





        if whichmodel == 'param_unet':
            model = unet.param_unet(input_size, filters, layers,Dropout_rate, whichloss)

        if whichmodel == 'unet':
            model = unet.unet(input_size, whichloss)


        if whichmodel == 'twolayernetwork':
            model = unet.twolayernetwork(input_size, kernel_size, Dropout_rate)

        if whichmodel == 'segnetwork':
            model = unet.segnetwork(input_size, kernel_size, Dropout_rate)





        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + '/' + whichmodel):
            os.makedirs(path + '/' + whichmodel)
        if not os.path.exists(path + '/' + whichmodel + '/' + whichloss):
            os.makedirs(path + '/' + whichmodel+ '/' + whichloss)
        if not os.path.exists(path + '/'+whichmodel+ '/' + whichloss+'/'+str(number_of_patients)+'patients'):
            os.makedirs(path + '/'+whichmodel+'/' + whichloss+'/'+ str(number_of_patients)+'patients')
        if not os.path.exists(path + '/'+whichmodel+'/' + whichloss+'/'+ str(number_of_patients)+'patients/' + str(layers)+'layers'):
            os.makedirs(path + '/'+whichmodel+'/' + whichloss+'/'+ str(number_of_patients)+'patients/' + str(layers)+'layers')
        save_dir = path + '/' + whichmodel +'/' + whichloss+'/' + str(number_of_patients)+'patients/'+ str(layers)+'layers'


        if whichmodel == 'param_unet' or whichmodel == 'unet':

            model_checkpoint = ModelCheckpoint(save_dir + '/unet.{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True, period=10)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='auto',
                                          baseline=None, restore_best_weights=False)
            history = model.fit(x_train, y_train, epochs=epochs, callbacks=[model_checkpoint, early_stopping], validation_split= validation_split_val)
        else:
            model.summary()
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='auto',
                                          baseline=None, restore_best_weights=False)
            if whichloss == 'dice':
                model.compile(loss=unet.dice_coef_loss,
                          optimizer='adam',
                          metrics=['accuracy'])
            if whichloss == 'binary_crossentropy':
                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

            history = model.fit(x_train, y_train, validation_split=validation_split_val, epochs=epochs, verbose=1, callbacks=[early_stopping])





        print(history.history.keys())



        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_accuracy_values.png'))

        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_loss_values.png'))

        plt.show()

        # model.save(os.path.join(save_dir, str(epochs) +'epochs_test.h5'))

        y_pred = []
        for i in x_test:
            i = np.expand_dims(i, -1)
            y_pred.append(model.predict(i, verbose = 1))

        np.save(os.path.join(save_dir, str(epochs) + 'epochs_y_pred'), y_pred)





        results = {
            "median_ROC_AUC_": "median_ROC_AUC",
            "number_of_patients" : number_of_patients,
            "model" : whichmodel,
            "epochs": epochs,
            "dice": "dice",
            "roc_auc": "roc_auc",
            "median_dice_score": "median_dice_score",
            "validation_split_val": validation_split_val,
            "unet_layers": layers,
            "filters": filters,
            "input_size": input_size,
            "loss": whichloss
        }
        dice = []
        roc_auc = []
        output = []

        for i in range(len(y_test)):
            output.append(np.squeeze(y_pred[i]))
            for s in range(y_test[i].shape[0]):

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
        results["epochs"] = epochs
        torch.save(results, os.path.join(save_dir, str(epochs) + 'epochs_evaluation_results'))

        print('DICE SCORE: ' + str(median_dice_score))
        print('ROC AUC:', str(median_ROC_AUC))

        methods.save_datavisualisation3(x_test, y_test, output, save_dir+'/' + str(layers)+'layers', True, True)

        all_results.append(results)


    best_idx = np.argmax([dict["median_dice_score"] for dict in all_results])

    print(' BEST MEDIAN DICE SCORE:', all_results[best_idx]["median_dice_score"], 'with', all_results[best_idx]["number_of_patients"],
          'number of patients, threshold =', ', epochs = ',
          all_results[best_idx]["epochs"])
