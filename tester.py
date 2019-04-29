

import numpy as np

import scoring_utils as su


y_test = np.load('y_test.npy')
y_pred = np.load('y_pred.npy')
y_test = np.array(y_test)
y_pred = np.array(y_pred)
n_x = 128
n_y = 128
mask_test = [[1 for x in range(0, n_x - 1)] for y in range(0, n_y -1)]
n_z = 1
n_subjects = 100
su.evaluate(y_pred, y_test, mask_test, n_subjects, n_x, n_y, n_z)