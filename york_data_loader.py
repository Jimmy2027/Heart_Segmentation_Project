import numpy as np
import imageio
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from keras.utils import np_utils
import os
import random
import scipy
import sklearn
from sklearn.feature_extraction import image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


"""
manual_seg_32points{z,t} contains a 65x2 vector of the segmentation at time frame t, 
and slice number z along the long axis. The first 32x2 values of the vector 
contain the endocardial landmarks forming the segmentation, and
elements 34:65x2 contain the epicardial segmentation. Row number 33
equals [0 0] to distinguish between the endocardial and epicardial contours. If no
segmentation exists at index z,t, manual_segs_32points{z,t}==-99999.

sol_yxzt(y,x,z,t) contains the pixel at row y, column x, taken from frame t and from
slice number z along the long axis.
"""

data_dir = 'Cardiac_MRI_dataset/manual_seg/manual_seg_32points_pat1.mat'

image = loadmat(data_dir)
#
# images = loadmat(data_dir, appendmat=True).get('IMAGES')
#
imgplot = plt.imshow(image)

plt.show()