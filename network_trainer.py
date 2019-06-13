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

whichlosses = ['binary_crossentropy']
# whichdatasets = ['York', 'ACDC']
whichdatasets = ['ACDC']


whichmodels = ['twolayernetwork']
# whichmodels = ['param_unet']



seeds = [1, 2, 3, 4] # for reproducibility
# seeds = [1]
batch_size = 32
# data_percs = [0.25, 0.5, 0.75, 1]  # between 0 and 1, not percentages
data_percs = [0.25]
# slice_percs = [0.25, 0.5, 0.75, 1]
slice_percs = [0.25]
filters = 64
splits = {1: (0.3, 0.1)}  # values for test and validation percentages

epochs = 1
threshold = 0.5

all_results = []

for whichdataset in whichdatasets:
    for whichloss in whichlosses:

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



        # total_numslices = np.concatenate(input[:]).shape[0]
        # num_trainslices = np.concatenate(data_dict[0]['1Perc']['Split#0']['train_images'][:]).shape[0]
        # num_valslices = np.concatenate(data_dict[0]['1Perc']['Split#0']['val_images'][:]).shape[0]
        # num_testslices = np.concatenate(data_dict[0]['1Perc']['Split#0']['test_images'][:]).shape[0]
        unet_input = []
        unet_labels = []

        """
        With this piece of code one can save the splits as images to visualize them
        """
        # for perc_index in range(len(data_percs)):
        #     for split_number in range(len(splits)):
        #
        #         data = data_dict[split_number][str(data_percs[perc_index]) + "Perc"]
        #
        #         x_train, y_train, x_val, y_val, x_test, y_test = \
        #             methods.get_datasets(data, split_number)
        #
        #         methods.save_datavisualisation2(x_train, y_train, 'York/training_data/split' + str(split_number )+ '/', True, True)
        #         methods.save_datavisualisation2(x_test, y_test, 'York/test_data/split' + str(split_number) + '/', True, True)
        #         methods.save_datavisualisation2(x_val, y_val, 'York/validation_data/split' + str(split_number) + '/', True, True)


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

                                input_size = x_train.shape[1:4]

                                print("****************************************************************************************************************")
                                print("EXPERIMENT:", layers, "layers,", perc * 100, "% total data,", slice_perc * 100, "% per pat,",
                                      whichloss, " as loss function,", "seed", split_number)
                                print("TRAIN DATA SIZE", x_train.shape[0])
                                print("VALIDATION DATA SIZE", x_val.shape[0])
                                # print("TEST DATA SIZE", x_test.shape[0])
                                print("****************************************************************************************************************")



                                x_train = np.expand_dims(x_train, -1)  #-1 for the last element
                                y_train = np.expand_dims(y_train, -1)
                                x_val = np.expand_dims(x_val, -1)
                                y_val = np.expand_dims(y_val, -1)



                                resolution = x_train.shape[1]

                                print(np.shape(x_train), np.shape(y_train))

                                # conv 2D default parameter: channels last: (batch, rows, cols, channels)

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
                                    os.makedirs(path + '/' + whichmodel + '/' + whichloss)
                                if not os.path.exists(path + '/'+whichmodel + '/' + whichloss+'/'+str(data_percs[perc_index])+'patients'):
                                    os.makedirs(path + '/'+ whichmodel+'/' + whichloss+'/'+ str(data_percs[perc_index])+'patients')

                                if not os.path.exists(path + '/'+whichmodel + '/' + whichloss+'/'+str(data_percs[perc_index])+'patients' + '/' + str(slice_perc) + 'slices'):
                                    os.makedirs(
                                        path + '/' + whichmodel + '/' + whichloss + '/' + str(data_percs[perc_index]) + 'patients'+ '/' + str(slice_perc) + 'slices')

                                if not os.path.exists(path + '/' + whichmodel+'/' + whichloss+'/'+ str(data_percs[perc_index]) +'patients/' +  '/' + str(slice_perc) + 'slices' + '/' + str(layers)+'layers'):
                                    os.makedirs(path + '/'+whichmodel+'/' + whichloss+'/'+ str(data_percs[perc_index])+'patients/'+  '/' + str(slice_perc) + 'slices' + '/' + str(layers)+'layers')

                                if not os.path.exists(path + '/' + whichmodel+'/' + whichloss+'/'+ str(data_percs[perc_index])+'patients/' +  '/' + str(slice_perc) + 'slices' + '/' + str(layers)+'layers/' + str(split_number) + 'split'):
                                    os.makedirs(path + '/'+whichmodel+'/' + whichloss+'/'+ str(data_percs[perc_index])+'patients/' +  '/' + str(slice_perc) + 'slices' + '/' +str(layers)+'layers/' + str(split_number) + 'split')

                                save_dir = path + '/' + whichmodel + '/' + whichloss + '/' + str(data_percs[perc_index]) + 'patients/' +  '/' + str(slice_perc) + 'slices' + '/' + str(layers) + 'layers/' + str(split_number) + 'split'
                                if data_augm==True:
                                    datagen = kp.image.ImageDataGenerator(
                                        featurewise_center=True,
                                        featurewise_std_normalization=True,
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True)
                                    # compute quantities required for featurewise normalization
                                    # (std, mean, and principal components if ZCA whitening is applied)
                                    datagen.fit(x_train)

                                if whichmodel == 'param_unet' or whichmodel == 'unet':

                                    model_checkpoint = ModelCheckpoint(save_dir + '/unet.{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True, period=100)
                                    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='auto',
                                                                  baseline=None, restore_best_weights=False)


                                    if data_augm ==True:
                                        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=epochs, callbacks=[model_checkpoint, early_stopping], validation_data=(x_val,y_val), batch_size =batch_size )
                                    else: history = model.fit(x_train, y_train, epochs=epochs, callbacks=[model_checkpoint, early_stopping], validation_data=(x_val, y_val), batch_size = batch_size)
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
                                    if data_augm == True:
                                        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / 32, validation_data=(x_val,y_val), epochs=epochs, verbose=1, callbacks=[early_stopping])
                                    else:
                                        history = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=epochs, verbose=1, callbacks=[early_stopping], batch_size = batch_size)






                                print(history.history.keys())


                                plt.figure()
                                # Plot training & validation accuracy values
                                plt.plot(history.history['acc'])
                                plt.plot(history.history['val_acc'])
                                plt.title('Model accuracy')
                                plt.ylabel('Accuracy')
                                plt.xlabel('Epoch')
                                plt.legend(['Train', 'Test'], loc='upper left')
                                plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_accuracy_values.png'))
                                plt.close()


                                plt.figure()
                                # Plot training & validation loss values
                                plt.plot(history.history['loss'])
                                plt.plot(history.history['val_loss'])
                                plt.title('Model loss')
                                plt.ylabel('Loss')
                                plt.xlabel('Epoch')
                                plt.legend(['Train', 'Validation'], loc='upper left')
                                plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_loss_values.png'))
                                plt.close()


                                # model.save(os.path.join(save_dir, str(epochs) +'epochs_test.h5'))

                                y_pred = []
                                for i in x_test:
                                    i = np.expand_dims(i, -1)
                                    y_pred.append(model.predict(i, verbose = 1))


                                np.save(os.path.join(save_dir, str(epochs) + 'epochs_y_pred'), y_pred)
                                np.save(os.path.join(save_dir, str(epochs) + 'epochs_y_test'), y_test)

                                results = {
                                    "median_ROC_AUC_": "median_ROC_AUC",
                                    "median_thresholded_dice": "median_thresholded_dice",
                                    "number_of_patients" : data_percs[perc_index],
                                    "model" : whichmodel,
                                    "epochs": epochs,
                                    "threshold": threshold,
                                    "dice": "dice",
                                    "roc_auc": "roc_auc",
                                    "median_dice_score": "median_dice_score",
                                    "unet_layers": layers,
                                    "filters": filters,
                                    "input_size": input_size,
                                    "which_split": split_number,
                                    "loss": whichloss,
                                    "nonthr_hausdorff": "nonthr_hausdorff",
                                    "median_nonthr_hausdorff": "median_nonthr_hausdorff",
                                    "thresholded_hausdorff": "thresholded_hausdorff",
                                "median_thresholded_hausdorff": "median_thresholded_hausdorff",
                                "slice_perc": slice_perc

                                }
                                dice = []
                                dice_thresholded = []
                                roc_auc = []
                                nonthr_hausdorff = []
                                thresholded_hausdorff = []
                                thresholded_y_pred = []
                                output = []

                                thresholded_output = []


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
                                        fpr, tpr, thresholds = roc_curve(y_true, y_pred_temp)
                                        roc_auc.append(auc(fpr, tpr))

                                median_ROC_AUC = np.median(roc_auc)
                                median_dice_score = np.median(dice)
                                median_thresholded_dice = np.median(dice_thresholded)
                                median_nonthr_hausdorff = np.median(nonthr_hausdorff)
                                median_thresholded_hausdorff = np.mean(thresholded_hausdorff)

                                results["median_dice_score"] = median_dice_score
                                results["median_thresholded_dice"] = median_thresholded_dice
                                results["median_ROC_AUC"] = median_ROC_AUC
                                results["dice"] = dice
                                results["roc_auc"] = roc_auc
                                results["epochs"] = epochs
                                results['median_nonthr_hausdorff'] = median_nonthr_hausdorff
                                results['median_thresholded_hausdorff'] = median_thresholded_hausdorff
                                torch.save(results, os.path.join(save_dir, str(epochs) + 'epochs_evaluation_results'))

                                print('DICE SCORE: ' + str(median_thresholded_dice))
                                print('ROC AUC:', str(median_ROC_AUC))
                                print('HAUSDORFF DISTANCE', str(median_thresholded_hausdorff))

                                methods.save_datavisualisation3(x_test, y_test, output, thresholded_output, str(round(median_thresholded_dice, 4)), save_dir+'/', True, True)

                                all_results.append(results)




                                plt.figure()
                                plt.hist(np.unique(y_pred[0]))
                                plt.title('mds: ' + str(round(results['median_thresholded_dice'], 4)) + '   ' + 'Hausdorff distance: ' + str(
                                    round(results['median_thresholded_hausdorff'], 4)))
                                plt.savefig(os.path.join(save_dir, str(epochs) + 'epochs_hist.png'))

best_idx = np.argmax([dict["median_dice_score"] for dict in all_results])
print(' BEST MEDIAN DICE SCORE:', all_results[best_idx]["median_dice_score"], 'with', all_results[best_idx]["number_of_patients"],
          'number of patients,', 'epochs = ',
          all_results[best_idx]["epochs"])

best_idx = np.argmax([dict["median_thresholded_dice"] for dict in all_results])
print(' BEST thresholded MEDIAN DICE SCORE:', all_results[best_idx]["median_thresholded_dice"], 'with', all_results[best_idx]["number_of_patients"],
          'number of patients,', 'epochs = ',
          all_results[best_idx]["epochs"])