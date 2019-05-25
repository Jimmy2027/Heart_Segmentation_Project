
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import os
import glob

basepath = 'ACDC_results/'
endfolder = False
modalites = os.listdir(basepath)


results = []
#
# def find_subdirs(path):
#     for folder in os.listdir(path):
#         if os.path.isdir(os.path.join(path, folder)):
#             path1 = os.path.join(path, folder)
#             find_subdirs(path1)
#         else:
#             if not folder.startswith('.'):
#                 objects = []
#                 for filename in os.scandir(path):
#                     objects.append(filename.name)
#                 for file in objects:
#                     if file.endswith(("_results")):
#                         result = torch.load(os.path.join(path, file))
#                         results.append(result)
#
#
# find_subdirs(basepath)
#
# best_idx = np.argmax([dict["median_dice_score"] for dict in results])
#
#
#
#
# # print(' BEST MEDIAN DICE SCORE:', results[best_idx]["median_dice_score"], 'with', results[best_idx]["number_of_patients"],
# #       'number of patients, threshold =', results[best_idx]["threshold"], ', epochs = ',
# #       results[best_idx]["epochs"])
#
# print(' BEST MEDIAN DICE SCORE:', results[best_idx]["median_dice_score"], 'with', results[best_idx]["number_of_patients"], 'number of patients, layers = ',results[best_idx]['unet_layers'] , 'epochs = ', results[best_idx]["epochs"], 'model = ', results[best_idx]['model'])



def read_dice_score(model, patients, layers):
    for folder in  os.listdir(os.path.join(basepath, model, str(patients)+'patients', str(layers)+'layers')):
        if folder.endswith('results'):
            print(folder, 'kufiufiztfiuzgouzgogz')
            results = torch.load(os.path.join(basepath, model, str(patients)+'patients', str(layers)+'layers',folder))






read_dice_score('unet', 118, 1)




something = 0