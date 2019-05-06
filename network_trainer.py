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



unet_input = np.load('unet_input.npy')
unet_labels = np.load('unet_labels.npy')

resolution = unet_input.shape[1]

x_train, x_test , y_train, y_test = model_selection.train_test_split(unet_input, unet_labels, test_size=0.3)

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




model, whichmodel = unet.unet(input_size)

if not os.path.exists('Results/' + whichmodel):
    os.makedirs('Results/' + whichmodel)
save_dir = 'Results/' + whichmodel

model_checkpoint = ModelCheckpoint(os.path.join(save_dir, str(epochs) +'epochs_unet.hdf5'), monitor='loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, validation_split=validation_split_val, epochs=epochs, callbacks=[model_checkpoint], verbose =1)





# model, whichmodel = unet.twolayernetwork(input_size, kernel_size, Dropout_rate)
# model.summary()
# model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
# history = model.fit(x_train, y_train, validation_split=validation_split_val, epochs=epochs, verbose =1)







print(history.history.keys())



# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig(os.path.join(save_dir,str(epochs) +'epochs_accuracy_values.png'))

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig(os.path.join(save_dir,str(epochs) +'epochs_loss_values.png'))

model.save(os.path.join(save_dir,str(epochs) +'epochs_test.h5'))


y_pred = model.predict(x_test)

np.save(os.path.join(save_dir,str(epochs) +'epochs_x_test'), x_test)


y_test = np.array(y_test)
y_pred = np.array(y_pred)
y_pred_shape = y_pred.shape
y_test_shape = y_test.shape
y_test = y_test.reshape(-1)
y_pred = y_pred.reshape(-1)
y_test = y_test//2
n_x = 128
n_y = 128
mask_test = [[1 for x in range(0, n_x - 1)] for y in range(0, n_y - 1)]
mask_test = np.array(mask_test)
n_z = 1
n_subjects = 30

#https://github.com/MonsieurWave/stroke-predict/blob/master/analysis/voxelWise/cv_framework.py
results = {
    # 'params': {
    #     'model_params': model_params,
    #     'rf': receptive_field_dimensions,
    #     'used_clinical': used_clinical,
    #     'masked_background': used_brain_masking,
    #     'scaled': feature_scaling,
    #     'settings_repeats': n_repeats,
    #     'settings_folds': n_folds,
    #     'settings_imgX_shape': imgX.shape,
    #     'settings_y_shape': y.shape,
    #     'failed_folds': failed_folds
    # },
    'train_evals': [],
    'test_accuracy': [],
    'test_roc_auc': [],
    'test_f1': [],
    'test_jaccard': [],
    'test_TPR': [],
    'test_FPR': [],
    'test_thresholded_volume_deltas': [],
    'test_unthresholded_volume_deltas': [],
    'test_image_wise_error_ratios': [],
    'test_image_wise_jaccards': [],
    # 'test_image_wise_hausdorff': [],
    # 'test_image_wise_modified_hausdorff': [],
    'test_image_wise_dice': [],
    'evaluation_thresholds': []
}

fold_result = su.evaluate(y_pred, y_test, mask_test, n_subjects, n_x, n_y, n_z)
results['test_accuracy'].append(fold_result['accuracy'])
results['test_f1'].append(fold_result['f1'])
results['test_roc_auc'].append(fold_result['roc_auc'])
results['test_TPR'].append(fold_result['tpr'])
results['test_FPR'].append(fold_result['fpr'])
results['test_jaccard'].append(fold_result['jaccard'])
results['test_thresholded_volume_deltas'].append(fold_result['thresholded_volume_deltas'])
results['test_unthresholded_volume_deltas'].append(fold_result['unthresholded_volume_deltas'])
results['test_image_wise_error_ratios'].append(fold_result['image_wise_error_ratios'])
results['test_image_wise_jaccards'].append(fold_result['image_wise_jaccards'])
# results['test_image_wise_hausdorff'].append(fold_result['image_wise_hausdorff'])
# results['test_image_wise_modified_hausdorff'].append(fold_result['image_wise_modified_hausdorff'])
results['test_image_wise_dice'].append(fold_result['image_wise_dice'])
# results['train_evals'].append(fold_result['train_evals'])
results['evaluation_thresholds'].append(fold_result['evaluation_threshold'])

y_test = y_test.reshape((y_test_shape))
y_pred = y_pred.reshape((y_pred_shape))




torch.save(results, os.path.join(save_dir,str(epochs) +'epochs_evaluation_results'))



threshold, upper, lower = results['evaluation_thresholds'], 1, 0
y_pred = np.where(y_pred > threshold, upper, lower)


np.save(os.path.join(save_dir,str(epochs) +'epochs_y_pred'), y_pred)



plt.figure(figsize=(resolution, resolution))

for i in range(0, 6):
    # plot original image
    ax = plt.subplot(2, 6, i + 1)
    plt.imshow(x_test[i,:,:,0], plt.cm.gray)

    # plot reconstructed image
    ax = plt.subplot(2, 6, 6 + i + 1)
    plt.imshow(y_pred[i,:,:,0], plt.cm.gray)

plt.savefig(os.path.join(save_dir,str(epochs) +'epochs_results.png'), bbox_inches='tight')

#imsave

something = 0
