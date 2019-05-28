
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import os
import glob

basepath = 'ACDC_results/'
endfolder = False
modalites = os.listdir(basepath)

def read_dice_score(model, lossfunction, patients, layers):
    for folder in  os.listdir(os.path.join(basepath, model, lossfunction, str(patients)+'patients', str(layers)+'layers')):
        if folder.endswith('results'):
            results = torch.load(os.path.join(basepath, model, lossfunction, str(patients)+'patients', str(layers)+'layers', folder))
        if folder.endswith('y_pred.npy'):
            y_pred = np.load(os.path.join(basepath, model, lossfunction, str(patients)+'patients', str(layers)+'layers', folder))
    return results, y_pred

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





# print(' BEST MEDIAN DICE SCORE:', results[best_idx]["median_dice_score"], 'with', results[best_idx]["number_of_patients"],
#       'number of patients, threshold =', results[best_idx]["threshold"], ', epochs = ',
#       results[best_idx]["epochs"])

print(' BEST MEDIAN DICE SCORE:', results[best_dice_idx]["median_dice_score"], 'with', results[best_dice_idx]["number_of_patients"], 'number of patients, layers = ',results[best_dice_idx]['unet_layers'] , 'epochs =', results[best_dice_idx]["epochs"], 'model =', results[best_dice_idx]['model'])
print(' BEST MEDIAN ROC AUC:', results[best_roc_idx]["median_dice_score"], 'with', results[best_roc_idx]["number_of_patients"], 'number of patients, layers = ',results[best_roc_idx]['unet_layers'] , 'epochs =', results[best_roc_idx]["epochs"], 'model =', results[best_roc_idx]['model'])


result, y_pred = read_dice_score(str(results[best_dice_idx]['model']), str(results[best_dice_idx]["loss"]), str(results[best_dice_idx]["number_of_patients"]), results[best_dice_idx]['unet_layers'])

plt.hist(np.unique(y_pred[0]))
plt.title('mds: ' + str(round(result['median_dice_score'], 4)) + '   ' + 'roc_auc: ' + str(round(result['median_ROC_AUC'], 4)) )
plt.show()

result,y_pred = read_dice_score(str(results[best_roc_idx]['model']), str(results[best_roc_idx]["loss"]), str(results[best_roc_idx]["number_of_patients"]), results[best_roc_idx]['unet_layers'])

# results, y_pred = read_dice_score('param_unet', 'dice', 118, 5)
# results, y_pred = read_dice_score('unet', 'binary_crossentropy', 118, 1)


plt.hist(np.unique(y_pred[0]))
plt.title('mds: ' + str(round(result['median_dice_score'], 4)) + '   ' + 'roc_auc: ' + str(round(result['median_ROC_AUC'], 4)) )
plt.show()

something = 0