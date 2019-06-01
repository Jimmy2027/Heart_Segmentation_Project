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
# whichloss = 'dice'
whichdataset = 'ACDC'
# whichdataset = 'York'
# whichmodel = 'param_unet'
# whichmodel = 'unet'
whichlosses = ['binary_crossentropy', 'dice']
# whichmodel = 'twolayernetwork'
whichmodels = ['segnetwork', 'param_unet', 'unet']



seeds = [1, 2, 3]
data_percs = [0.25, 0.5, 0.75, 1]  # between 0 and 1, not percentages
filters = 64
splits = {1: (0.3, 0.1), 2: (0.3, 0.1), 3: (0.3, 0.1)}  # values for test and validation percentages
# layers_arr = [5,4,3,2]

# layers_arr = [1]
epochs = 1


all_results = []



if whichdataset == 'York':
    input = np.load('YCMRI_128x128_images.npy', allow_pickle=True)
    labels = np.load('YCMRI_128x128_labels.npy', allow_pickle=True)
    path = 'York_results'
    plt.figure()
    for i in range(len(labels)):
        plt.hist(np.unique(labels[i][:]))

    plt.show()


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

    plt.figure()
    for i in range(len(labels)):
        plt.hist(np.unique(labels[i][:]))

    plt.show()

data_dict = []
for seed in seeds:
    data_dict.append(methods.getalldata(input, labels, data_percs, splits, seed))

unet_input = []
unet_labels = []

for whichloss in whichlosses:
    for whichmodel in whichmodels:
        if whichmodel == 'param_unet':
            layers_arr = [5, 4, 3, 2]
        else:
            layers_arr = [1]
        for layers in layers_arr:

            for perc_index in range(len(data_percs)):
                for split_number in range(len(splits)):
                    data = data_dict[split_number][str(data_percs[perc_index]) + "Perc"]

                    x_train, y_train, x_val, y_val, x_test, y_test = \
                        methods.get_datasets(data, split_number)



                    x_train = np.concatenate(x_train, axis = 0)
                    y_train = np.concatenate(y_train, axis = 0)

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
                        os.makedirs(path + '/' + whichmodel + '/' + whichloss)
                    if not os.path.exists(path + '/'+whichmodel + '/' + whichloss+'/'+str(data_percs[perc_index])+'patients'):
                        os.makedirs(path + '/'+ whichmodel+'/' + whichloss+'/'+ str(data_percs[perc_index])+'patients')
                    if not os.path.exists(path + '/' + whichmodel+'/' + whichloss+'/'+ str(data_percs[perc_index])+'patients/' + str(layers)+'layers'):
                        os.makedirs(path + '/'+whichmodel+'/' + whichloss+'/'+ str(data_percs[perc_index])+'patients/' + str(layers)+'layers')

                    if not os.path.exists(path + '/' + whichmodel+'/' + whichloss+'/'+ str(data_percs[perc_index])+'patients/' + str(layers)+'layers/' + str(split_number) + 'split'):
                        os.makedirs(path + '/'+whichmodel+'/' + whichloss+'/'+ str(data_percs[perc_index])+'patients/' + str(layers)+'layers/' + str(split_number) + 'split')

                    save_dir = path + '/' + whichmodel + '/' + whichloss + '/' + str(data_percs[perc_index]) + 'patients/' + str(layers) + 'layers/' + str(split_number) + 'split'


                    if whichmodel == 'param_unet' or whichmodel == 'unet':

                        model_checkpoint = ModelCheckpoint(save_dir + '/unet.{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True, period=2000)
                        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='auto',
                                                      baseline=None, restore_best_weights=False)
                        tensorboard = keras.callbacks.TensorBoard(log_dir=os.path.join(save_dir, '/logs'), histogram_freq=10, batch_size=32, write_graph=True,
                                                    write_grads=True, write_images=True, embeddings_freq=0,
                                                    embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                                    update_freq='epoch')

                        history = model.fit(x_train, y_train, epochs=epochs, callbacks=[model_checkpoint, early_stopping, tensorboard], validation_data=(x_val,y_val))
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





                    results = {
                        "median_ROC_AUC_": "median_ROC_AUC",
                        "number_of_patients" : data_percs[perc_index],
                        "model" : whichmodel,
                        "epochs": epochs,
                        "dice": "dice",
                        "roc_auc": "roc_auc",
                        "median_dice_score": "median_dice_score",
                        "validation_split_val": validation_split_val,
                        "unet_layers": layers,
                        "filters": filters,
                        "input_size": input_size,
                        "which_split": split_number,
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

                    methods.save_datavisualisation3(x_test, y_test, output, str(round(median_dice_score, 4)), save_dir+'/', True, True)

                    all_results.append(results)

                    torch.save(y_test, os.path.join(save_dir, 'y_test'))



                plt.figure()
                plt.hist(np.unique(y_pred[0]))
                plt.title('mds: ' + str(round(results['median_dice_score'], 4)) + '   ' + 'roc_auc: ' + str(
                    round(results['median_ROC_AUC'], 4)))
                plt.savefig(os.path.join(save_dir, str(epochs) + 'epochs_hist.png'))

best_idx = np.argmax([dict["median_dice_score"] for dict in all_results])
print(' BEST MEDIAN DICE SCORE:', all_results[best_idx]["median_dice_score"], 'with', all_results[best_idx]["number_of_patients"],
          'number of patients,', 'epochs = ',
          all_results[best_idx]["epochs"])

