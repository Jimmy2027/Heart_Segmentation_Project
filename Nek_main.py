from models import param_unet, segnetwork
from Nek_functions import *
from matplotlib import pyplot as plt
import time
from medpy.metric.binary import hd
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import os
import methods
from email_notification import NotificationSystem

notification_system = NotificationSystem()
title = 'testing_data_augmentation'


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
import tensorflow as tf
from keras import backend as K

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    #print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))





#************************** Fixed Parameters *****************************
filters = 64
dropout_rate = 0.5
whichmodels = ['param_unet']
split = (0.3,0.1)
basic_batch_size = 24
val_loss_patience = 15
gray_images_patience = 1
amount_of_non_gray_predictions = 4
min_val_loss = 1e-3

#**************************** Main Parameters *****************************
testing = True

if testing:
    maxepochs = 1
    patient_percs = [0.1]
    levels_arr = [1]
    slice_percs = [0.1]
else:
    maxepochs = 500
    patient_percs = [0.25, 0.5, 0.75, 1]
    levels_arr = [4]
    slice_percs = [0.25, 0.5, 0.75, 1]
datasets = ['ACDC']
data_augm = False
whichloss = 'binary_crossentropy'

#************************** Data Augmentation Parameters *************************************
single_param = False # tests all data augm param one by one ... only works if data_augm also True
rotation_range = 30
width_shift_range = 0.2
height_shift_range = 0.2
zoom_range = 0.2
horizontal_flip = 1
vertical_flip = 1
data_gen_args_list = get_args_list(data_augm, single_param, rotation_range, width_shift_range, height_shift_range, zoom_range, horizontal_flip, vertical_flip)




times = []

for whichmodel in whichmodels:
    if whichmodel == 'segnet':
        levels_arr = [0]
    for dataset in datasets:
        if dataset == 'York':
            input = np.load('YCMRI_128x128_images.npy', allow_pickle=True)
            labels = np.load('YCMRI_128x128_labels.npy', allow_pickle=True)
            if data_augm == True:
                path = 'York_results_data_augm'
            else:
                path = 'York_results'

            # plt.figure()
            # for i in range(len(labels)):
            #     plt.hist(np.unique(labels[i][:]))
            #
            # plt.show()

        if dataset == 'ACDC':

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

        for levels in levels_arr:
            if levels > 4:
                batch_size = int((0.5**(levels-4))*basic_batch_size)
            else:
                batch_size = basic_batch_size
            for patient_perc in patient_percs:
                if patient_perc != 1:
                    slice_percs = [1]
                else:
                    if not testing:
                        slice_percs = [1]
                    else:
                        slice_percs = [0.25]
                for slice_perc in slice_percs:
                    for args in data_gen_args_list:
                        non_gray_images = 0
                        seed = 0
                        while non_gray_images != amount_of_non_gray_predictions: # make sure (amount_of_splits) non-gray images are saved
                            seed += 1

                            total_start_time = time.time()

                            # ****************************** Seed all sources of randomness ***************************
                            reset_keras()
                            random.seed(seed)
                            np.random.seed(seed)
                            tf.set_random_seed(seed)
                            os.environ['PYTHONHASHSEED'] = str(seed)



                            # *********************************** Get Data ****************************
                            train_pats, test_pats, val_pats = get_patient_split(len(input), split)

                            train_pats = get_total_perc_pats(train_pats, patient_perc)
                            test_pats = get_total_perc_pats(test_pats, 1)
                            val_pats = get_total_perc_pats(val_pats, 1)

                            x_train, y_train = get_slice_perc_split(input, labels, train_pats, slice_perc)
                            if testing == True:
                                x_test, y_test = get_slice_perc_split(input, labels, test_pats, slice_perc)
                                x_val, y_val = get_slice_perc_split(input, labels, val_pats, slice_perc)
                            else:
                                x_test, y_test = get_slice_perc_split(input, labels, test_pats, 1)
                                x_val, y_val = get_slice_perc_split(input, labels, val_pats, 1)

                            input_size = x_train.shape[1:4]

                            print(
                                "****************************************************************************************************************")
                            print("EXPERIMENT:", levels, "levels,", patient_perc * 100, "% total data,", slice_perc * 100,
                                  "% per pat,",
                                  whichloss, " as loss function,", "seed", split, 'on', dataset)
                            print("TRAIN DATA SIZE", x_train.shape[0])
                            print("VALIDATION DATA SIZE", x_val.shape[0])
                            # print("TEST DATA SIZE", x_test.shape[0])
                            print(
                                "****************************************************************************************************************")

                            train_images = np.expand_dims(x_train, -1)  # -1 for the last element
                            train_masks = np.expand_dims(y_train, -1)
                            val_images = np.expand_dims(x_val, -1)
                            val_masks = np.expand_dims(y_val, -1)
                            test_images = np.expand_dims(np.array(x_test), -1)
                            test_masks = np.expand_dims(np.array(y_test), -1)
                            prefix = ''
                            if data_augm:
                                train_generator, x_augm_save_dir, prefix, augm_count_before = get_train_generator(
                                    args, train_images, train_masks, val_images, val_masks, seed, single_param,
                                    batch_size, dataset)

                            print("****************************************************************************************************************")
                            print("EXPERIMENT:", levels, "levels,", patient_perc*100, "% total data,", slice_perc*100, "% per pat,", "seed", seed)
                            print("TRAIN DATA SIZE", train_images.shape[0])
                            print("VALIDATION DATA SIZE", val_images.shape[0])
                            print("TEST DATA SIZE", test_images.shape[0])
                            print("****************************************************************************************************************")

                            # *********************** Setup model ********************

                            input_size = train_images.shape[1:4]

                            if whichmodel == "param_unet":
                                model = param_unet(input_size, filters, levels, dropout_rate, whichloss)

                            if whichmodel == "segnet":
                                model = segnet(input_size, 3, dropout_rate, whichloss)


                            # *********************************** Train Model **********************************************

                            gray_image = False
                            gray_counter = 0
                            loss_counter = 0
                            best_val_loss = 1
                            acc = []
                            val_acc = []
                            loss = []
                            val_loss = []
                            for i in range(maxepochs):
                                print("-------------------- Epoch #", i+1,"--------------------")
                                start_time = time.time()
                                if data_augm:
                                    history = model.fit_generator(train_generator, epochs=1,
                                                                  steps_per_epoch=train_images.shape[0] // batch_size,
                                                                  validation_data=(val_images, val_masks), verbose=0,
                                                                  shuffle=True)
                                else:
                                    history = model.fit(train_images, train_masks, epochs=1, batch_size=batch_size,
                                                        validation_data=(val_images, val_masks), verbose=0,
                                                        shuffle=True)

                                print("Finished training:", "val_acc", round(history.history['val_acc'][0],4), ", val_loss", round(history.history['val_loss'][0],4))

                                # ***************************************** Save data for plots later ********************************************

                                acc.append(history.history['acc'][0])
                                val_acc.append(history.history['val_acc'][0])
                                loss.append(history.history['loss'][0])
                                val_loss.append(history.history['val_loss'][0])

                                # ********************* Early Stopping if no improvement bigger than min_val_loss after val_loss_patience ******************************

                                new_val_loss = round(history.history['val_loss'][0],4)
                                diff = round(best_val_loss - new_val_loss,4)
                                if diff > min_val_loss:
                                    print("New best loss. Improvement of", -diff)
                                    best_val_loss = new_val_loss
                                    loss_counter = 0
                                else:
                                    loss_counter += 1
                                    print("No best loss since", loss_counter, "epochs")
                                    if loss_counter == val_loss_patience:
                                        print("Too many epochs without validation loss improvement")
                                        epochs = i+1
                                        print("---------------------------------------------------")
                                        break

                                # ******************* Early Stopping if difference between max and min value of prediction less than 0.1 (=gray image) after gray_image_patience **************************

                                test_image = test_images[0,:,:]
                                test_image = np.expand_dims(test_image, 0)

                                mask_prediction = model.predict(test_image, verbose=0)

                                minima = []
                                maxima = []
                                for l in range(mask_prediction.shape[1]):
                                    minima.append(min(mask_prediction[0, l, :]))
                                    maxima.append(max(mask_prediction[0, l, :]))
                                if (max(maxima) - min(minima) >= 0.1):  # only let's non-gray images through
                                    gray_counter = 0
                                    print("Time passed:", str(round(time.time() - start_time,2)), "seconds")
                                    continue
                                else:
                                    gray_counter += 1
                                    print("Gray Image detected", gray_counter, "in a row")
                                    print("Time passed:", str(round(time.time() - start_time,2)), "seconds")
                                    if gray_counter == gray_images_patience:
                                        gray_image = True
                                        print("Too many gray images")
                                        print("---------------------------------------------------")
                                        epochs = i + 1
                                        break


                            # ************************** Continue only if image is not gray *********************************

                            if not gray_image:
                                non_gray_images += 1

                                if data_augm:
                                    augm_count_after = add_count(x_augm_save_dir)
                                    augm_count = augm_count_after - augm_count_before

                                if testing == True:
                                    path = "temp/new/" + \
                                           whichmodel + '-' + \
                                           str(epochs) + "_epochs" + '-' + \
                                           str(int(patient_perc * 100)) + '%_total_data-' + \
                                           str(int(slice_perc * 100)) + '%_per_pat-(' + \
                                           str(train_images.shape[0]) + '_images)-' + \
                                           str(levels) + '_levels' + '-' + 'seed_' + \
                                           str(seed) + '-'
                                else:
                                    path = dataset + "_results_data_augm/new/" + \
                                           whichmodel + '-' + \
                                           str(epochs) + "_epochs" + '-' + \
                                           str(int(patient_perc * 100)) + '%_total_data-' + \
                                           str(int(slice_perc * 100)) + '%_per_pat-('+\
                                           str(train_images.shape[0])+'_images)-' + \
                                           str(levels) + '_levels' + '-' + 'seed_' + \
                                           str(seed) + '-'

                                if data_augm:
                                    path += prefix + '(' + \
                                            str(augm_count) + '_augm)-'
                                else: path += 'unaugm-'


                                index = 1
                                if not os.path.exists(path + str(index) + '/'):
                                    os.makedirs(path + str(index) + '/')
                                    save_dir = path + str(index) + '/'
                                else:
                                    while os.path.exists(path + str(index) + '/'):
                                        index += 1
                                    os.makedirs(path + str(index) + '/')
                                    save_dir = path + str(index) + '/'



                                # Plot training & validation accuracy values

                                x = range(1,epochs+1)

                                plt.plot(x, acc)
                                plt.plot(x, val_acc)
                                plt.title('Model accuracy')
                                plt.ylabel('Accuracy')
                                plt.xlabel('Epoch')
                                plt.legend(['Train', 'Test'], loc='upper left')
                                plt.ylim(0, 1)
                                plt.savefig(os.path.join(save_dir, str(epochs) + 'epochs_accuracy_values.png'))
                                # plt.show()

                                plt.close()

                                # Plot training & validation loss values
                                plt.plot(x, loss)
                                plt.plot(x, val_loss)
                                plt.title('Model loss')
                                plt.ylabel('Loss')
                                plt.xlabel('Epoch')
                                plt.legend(['Train', 'Test'], loc='upper left')
                                plt.ylim(0, 1)
                                plt.savefig(os.path.join(save_dir, str(epochs) + 'epochs_loss_values.png'))
                                # plt.show()

                                plt.close()

                                mask_prediction = model.predict(test_images, verbose=1)

                                model.save(os.path.join(save_dir, str(epochs) + 'epochs_test.h5'))

                                rounded_pred = threshold(mask_prediction, 0.5, 0.5)
                                rounded_pred = np.squeeze(rounded_pred, 3)
                                mask_prediction = np.array(mask_prediction)
                                mask_prediction = np.squeeze(mask_prediction, 3)

                                np.save(os.path.join(save_dir, str(epochs) + 'epochs_mask_prediction'), mask_prediction)
                                np.save(os.path.join(save_dir, str(epochs) + 'test_images'), test_images)
                                np.save(os.path.join(save_dir, str(epochs) + 'test_masks'), test_masks)
                                np.save(os.path.join(save_dir, str(epochs) + 'rounded_mask_pred'), rounded_pred)



                                rounded_dice = []
                                dice = []
                                thresholded_hausdorff = []
                                hausdorff = []
                                for s in range(test_masks.shape[0]):
                                        rounded_dice.append(getdicescore(test_masks[s, :, :, 0], rounded_pred[s, :, :]))
                                        dice.append(getdicescore(test_masks[s, :, :, 0], mask_prediction[s, :, :]))
                                        if np.max(rounded_pred[s, :, :]) != 0:
                                            thresholded_hausdorff.append(hd(rounded_pred[s, :, :], test_masks[s, :, :, 0]))
                                        if np.max(rounded_pred[s, :, :]) != 0:
                                            hausdorff.append(hd(rounded_pred[s, :, :], test_masks[s, :, :, 0]))

                                median_rounded_dice_score = np.median(rounded_dice)
                                median_thresholded_hausdorff = np.mean(thresholded_hausdorff)

                                median_thresholded_hausdorff = round(median_thresholded_hausdorff, 2)
                                median_rounded_dice_score = round(median_rounded_dice_score, 3)



                                #save_visualisation2(train_images[:, :, :, 0], train_masks[:, :, :, 0], median_rounded_dice_score, median_thresholded_hausdorff, save_dir)
                                save_visualisation(test_images[:,:,:,0], test_masks[:,:,:,0], mask_prediction, rounded_pred,
                                                   median_rounded_dice_score, median_thresholded_hausdorff, prefix, save_dir)




                                print("****************************************************************************************************************")
                                print('DICE SCORE:', str(median_rounded_dice_score))
                                print('HAUSDORFF DISTANCE:', str(median_thresholded_hausdorff))
                                elapsed_time = int(time.time()-total_start_time)
                                hours = elapsed_time//3600
                                minutes = elapsed_time//60 - hours*60
                                seconds = elapsed_time - hours*3600 - minutes*60
                                print('ELAPSED TIME:', hours,"hours", minutes, "minutes", seconds, "seconds")
                                print("****************************************************************************************************************")
                                times.append((patient_perc, "% of patients,", slice_perc, "% of slices,", levels, "levels:", elapsed_time))
                                if data_augm:
                                    print("AUGMENTED DATA SIZE:", augm_count)
                                    print("****************************************************************************************************************")
print(times)
body = title
if testing ==  False:
    notification_system.send_message(title, body)