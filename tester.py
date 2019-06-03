from medpy.metric.binary import hd, dc

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import os
import glob
import scoring_utils as su


basepath = 'York_results1/'
endfolder = False
modalites = os.listdir(basepath)

def read_dice_score(model, lossfunction, patients, layers, split):
    for folder in  os.listdir(os.path.join(basepath, model, lossfunction, str(patients)+'patients', str(layers)+'layers', str(split) + 'split')):
        if folder.endswith('results'):
            results = torch.load(os.path.join(basepath, model, lossfunction, str(patients)+'patients', str(layers)+'layers', str(split) + 'split', folder))
        if folder.endswith('y_pred.npy'):
            y_pred = np.load(os.path.join(basepath, model, lossfunction, str(patients)+'patients', str(layers)+'layers', str(split) + 'split', folder))
    return results, y_pred



def compute_dice_score(model, lossfunction, patients, layers, split):
    for folder in os.listdir(
        os.path.join(basepath, model, lossfunction, str(patients) + 'patients', str(layers) + 'layers', str(split) + 'split')):
        if folder.endswith('y_pred.npy'):
            y_pred = np.load(os.path.join(basepath, model, lossfunction, str(patients) + 'patients', str(layers) + 'layers', str(split) + 'split', folder))
        if folder.endswith('y_test.npy'):
            y_test = np.load(os.path.join(basepath, model, lossfunction, str(patients) + 'patients', str(layers) + 'layers', str(split) + 'split', folder))

    dice = []
    output = []
    dice_threshold = []
    threshold = 0.5
    for i in range(len(y_test)):
        output.append(np.squeeze(y_pred[i]))
        for s in range(y_test[i].shape[0]):
            dice.append(dc(output[i][s], y_test[i][s]))
            dice_threshold.append(dc(np.where(output[i][s] > threshold, 1, 0), y_test[i][s]))

    median_dice_score = np.median(dice)
    median_thrdice_score = np.median(dice_threshold)

    plt.figure()
    plt.hist(np.unique(y_pred[0]))
    plt.title('mds: ' + str(round(median_dice_score, 4)))
    plt.show()

    plt.figure()
    plt.plot(y_pred[0][:,:,0])
    plt.show()

    plt.figure()
    plt.plot(np.where(y_pred[0][:,:,0] > threshold, 1, 0))
    plt.show()

    plt.figure()
    plt.hist(np.unique(np.where(y_pred[0] > threshold, 1, 0)))
    plt.title('med thr dice ' + str(
        round(median_thrdice_score, 4)))
    plt.show()

    return median_dice_score, median_thrdice_score

def print_best_scores():
    results = []

    def find_subdirs(path):
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                path1 = os.path.join(path, folder)
                find_subdirs(path1)
            else:
                if not folder.startswith('.'):
                    objects = []
                    for filename in os.scandir(path):
                        objects.append(filename.name)
                    for file in objects:
                        if file.endswith(("_results")):
                            result = torch.load(os.path.join(path, file))
                            results.append(result)


    find_subdirs(basepath)

    best_dice_idx = np.argmax([dict["median_dice_score"] for dict in results])
    best_roc_idx = np.argmax([dict["median_ROC_AUC"] for dict in results])





    print(' BEST MEDIAN DICE SCORE:', results[best_dice_idx]["median_dice_score"], 'with', results[best_dice_idx]["number_of_patients"], 'number of patients, layers = ',results[best_dice_idx]['unet_layers'] , ', epochs =', results[best_dice_idx]["epochs"], ', model =', results[best_dice_idx]['model'], 'and split nr. ', results[best_dice_idx]['which_split'])
    print(' BEST MEDIAN ROC AUC:', results[best_roc_idx]["median_dice_score"], 'with', results[best_roc_idx]["number_of_patients"], 'number of patients, layers = ',results[best_roc_idx]['unet_layers'] , ', epochs =', results[best_roc_idx]["epochs"], ', model =', results[best_roc_idx]['model'],'and split nr. ', results[best_roc_idx]['which_split'])


    result, y_pred = read_dice_score(str(results[best_dice_idx]['model']), str(results[best_dice_idx]["loss"]), str(results[best_dice_idx]["number_of_patients"]), results[best_dice_idx]['unet_layers'], results[best_dice_idx]['which_split'])

    plt.hist(np.unique(y_pred[0]))
    plt.title('mds: ' + str(round(result['median_dice_score'], 4)) + '   ' + 'roc_auc: ' + str(round(result['median_ROC_AUC'], 4)) )
    plt.show()

    result,y_pred = read_dice_score(str(results[best_roc_idx]['model']), str(results[best_roc_idx]["loss"]), str(results[best_roc_idx]["number_of_patients"]), results[best_roc_idx]['unet_layers'],results[best_roc_idx]['which_split'])

    # results, y_pred = read_dice_score('param_unet', 'dice', 118, 5)
    # results, y_pred = read_dice_score('unet', 'binary_crossentropy', 118, 1)


    plt.hist(np.unique(y_pred[0]))
    plt.title('mds: ' + str(round(result['median_dice_score'], 4)) + '   ' + 'roc_auc: ' + str(round(result['median_ROC_AUC'], 4)) )
    plt.show()






# median_dice_score, median_thrdice_score = compute_dice_score('param_unet', 'binary_crossentropy', 1, 3, 0)
print_best_scores()
# results, y_pred = read_dice_score('twolayernetwork', 'binary_crossentropy', 0.25, 1, 0)
something = 0