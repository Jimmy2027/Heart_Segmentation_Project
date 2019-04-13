import scipy.io
import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib
import os
import numpy as np
from nilearn import image
from nilearn.input_data import NiftiMasker
import metrics_acdc
import collections
from collections import Counter
from scipy import ndimage as nd




def find_middle(img_data):  #TODO here I am taking the middle of the image, what I want is to have the middle of the segmentation though
    """

    :param img_data: single image data
    :return: center coordinate of input image
    """

    n_i, n_j, n_k = img_data.shape
    center_i = (n_i - 1) // 2  # // for integer division
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2

    img_center = [center_i, center_j, center_k]
    # print('Image center: ', center_i, center_j, center_k)
    center_vox_value = img_data[center_i, center_j, center_k]
    # print('Image center value: ', center_vox_value)

    return img_center


def plot_img_with_label(labels, images):
    """
    Plots 50 images with labels
    :param labels: list with label paths
    :param images: list with images paths
    """
    for i in range(1, 50):
        plotting.plot_roi(labels[i], bg_img=images[i], cmap='Paired')
        # plotting.plot_img(i)
        plotting.show()

def crop_images(cropper_size, center_of_masses, img_data, myocar_labels):
    """
    Data shape: (216, 256, 10)
    :param average_i:
    :param average_j:
    :param average_k:
    :param img_data: data_array of to be cropped image
    :param myocar_labels: data_array of to be copped labels
    :return: data arrays of cropped images and labels
    """
    print('len img data:', len(img_data))
    print('len myo labels', len(myocar_labels))
    min_layer = 8
    data_shape = img_data[0].shape
    cropped_img_data = np.empty((len(img_data), 100, 100, min_layer))
    cropped_myocar_labels = np.empty((len(myocar_labels), 100, 100, min_layer))
    for t, i in enumerate(img_data):
        print('t:',t)
        print('i shape:', i.shape)
        ishape = i.shape
        if ishape[2] > min_layer:     #if image has more layers than usual, remove them
            i = i[:,:,0:min_layer]
            print('new i shape:', i.shape)

        if ishape[0] != data_shape[0] or ishape[1] != data_shape[1]:
            i = np.empty((100, 100, min_layer))
            print('new i shape:', i.shape)
        cropped_img_data[t] = i[average_i - cropper_size: average_i + cropper_size, average_j - cropper_size:average_j + cropper_size, :]

    for t, i in enumerate(myocar_labels):
        ishape = i.shape
        if ishape[2] > min_layer:     #if image has more layers than usual, remove them
            i = i[:,:,0:min_layer]
            print('new i shape:', i.shape)

        if ishape[0] != data_shape[0] or ishape[1] != data_shape[1]:
            i = np.empty((100, 100, min_layer))
            print('new i shape:', i.shape)

        cropped_myocar_labels[t] = i[average_i - cropper_size: average_i + cropper_size, average_j - cropper_size: average_j + cropper_size, :]

    return cropped_img_data, cropped_myocar_labels

def crop_images2(cropper_size, data_shape, data_length, average_i, average_j, average_k, img_data, myocar_labels):
    """
    This function only crops images with the data shape (216, 256, 10)
    :param average_i:
    :param average_j:
    :param average_k:
    :param img_data: data_array of to be cropped image
    :param myocar_labels: data_array of to be copped labels
    :return: data arrays of cropped images and labels
    """
    print('len img data:', len(img_data))
    print('len myo labels', len(myocar_labels))
    data_shape = img_data[0].shape
    cropped_img_data = np.empty((data_length, 100, 100, 10))
    cropped_myocar_labels = np.empty((len(myocar_labels), 100, 100, 10))
    t_img = 0
    t_label = 0
    for i in img_data:
        if i.shape == data_shape:

            cropped_img_data[t_img] = i[average_i - cropper_size: average_i + cropper_size, average_j - cropper_size:average_j + cropper_size, :]
            t_img= t_img+1

    for i in myocar_labels:
        if i.shape == data_shape:

            cropped_myocar_labels[t_label] = i[average_i - cropper_size: average_i + cropper_size, average_j - cropper_size: average_j + cropper_size, :]
            t_label = t_label + 1

    return cropped_img_data, cropped_myocar_labels
