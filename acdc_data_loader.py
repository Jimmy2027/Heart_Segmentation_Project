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





def data_normalization(data):

    for i in data:
        i = i - min(i)
        i = i / max(i)
        np.clip(i, 0, np.percentile(i, 99))
    return data



def remove_other_segmentations(labels):
    """
    removes all other segmentations apart from the myocardium one

    :param label: list of strings with paths to labels
    :return: array with all the myocar labels

        https://nipy.org/nibabel/images_and_memory.html

    """
    myocar_data = []
    for i in labels:
        img = nib.load(i)
        img_data = img.get_data()
        img_data[img_data != 2] = 0
        myocar_data.append(img_data)
    return myocar_data

#TODO load image data as well as data paths

def load_images(data_dir, raw_image_path01, raw_image_path12, label_path01, label_path12):
    """

    :param data_dir, both paths to systole and diastole images with corresponding labels
    :return: array of paths to unlabeled images and array with path to corresponding labels
    """
    im_data = []
    labels = []
    for o in os.listdir(data_dir): #o: patientxxx
        # print('o: ', o)
        if not o.startswith('.') and os.path.isdir(os.path.join(data_dir, o)):

            for x in os.listdir(os.path.join(data_dir, o)): #x: patientXXX_frameXX.nii.gz

                if not x.startswith('.'):
                    # print('x: ' , data_dir + '/' + x)
                    # print('path: ', os.path.join(data_dir, o) + raw_image_path01)
                    # print(data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path01) or (
                    #         data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path12)
                    if (data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path01) or (
                            data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path12):
                        im_data.append(data_dir + '/' + o + '/'+ x)

                    if data_dir + '/' + x == (os.path.join(data_dir, o) + label_path01) or data_dir + '/' + x == (
                            os.path.join(data_dir, o) + label_path12):

                        labels.append(data_dir + '/' + o + '/' + x)

    return im_data, labels

def load_data(data_paths):
    data = []
    for i in data_paths:
        img = nib.load(i)
        img_data = img.get_data()
        data.append(img_data)

    return data


def display(img_data, block = True):
    """
    Plots middle slice of 3D image

    :param img_data: data to be plotted
    :param block: if programm should be blocked

    """
    def show_slices(slices):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")

    n_i, n_j, n_k = img_data.shape
    center_i = (n_i - 1) // 2  # // for integer division
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2
    # print('Image center: ', center_i, center_j, center_k)
    center_vox_value = img_data[center_i, center_j, center_k]
    # print('Image center value: ', center_vox_value)

    slice_0 = img_data[center_i, :, :]
    slice_1 = img_data[:, center_j, :]
    slice_2 = img_data[:, :, center_k]

    show_slices([slice_0, slice_1, slice_2])

    plt.suptitle("Center slices for image")
    plt.show(block = block)


def find_center_of_mass(data):
    return nd.measurements.center_of_mass(data)




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


def crop_images(cropper_size, average_i, average_j, average_k, img_data, myocar_labels):        #TODO ATTENTION: data shape not uniform... how can I uniformise the shapes?
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
        if ishape[2] > min_layer:     #if image has more layers than usual, remove them      TODO is this okay?
            i = i[:,:,0:min_layer]
            print('new i shape:', i.shape)

        if ishape[0] != data_shape[0] or ishape[1] != data_shape[1]:
            i = np.empty((100, 100, min_layer))
            print('new i shape:', i.shape)
        cropped_img_data[t] = i[average_i - cropper_size: average_i + cropper_size, average_j - cropper_size:average_j + cropper_size, :]

    for t, i in enumerate(myocar_labels):
        ishape = i.shape
        if ishape[2] > min_layer:     #if image has more layers than usual, remove them      TODO is this okay?
            i = i[:,:,0:min_layer]
            print('new i shape:', i.shape)

        if ishape[0] != data_shape[0] or ishape[1] != data_shape[1]:
            i = np.empty((100, 100, min_layer))
            print('new i shape:', i.shape)

        cropped_myocar_labels[t] = i[average_i - cropper_size: average_i + cropper_size, average_j - cropper_size: average_j + cropper_size, :]

    return cropped_img_data, cropped_myocar_labels


def crop_images2(cropper_size, data_shape, data_length, average_i, average_j, average_k, img_data, myocar_labels):        #TODO ATTENTION: data shape not uniform... how can I uniformise the shapes?
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

def find_most_common_shape2(data):
    """

    :param data: data Array
    :return: list of most common element shape in array and number of occurences
    """
    temp = []
    for t, i in enumerate(data):
        temp.append(i.shape)
    b = Counter(temp)
    return b.most_common(1)

def find_most_common_shape(data):
    """

    :param data: data Array
    :return: list of most common element shape in array and number of occurences
    """
    temp = np.empty((len(data)))
    for t, i in enumerate(data):
        temp[t] = i.shape
    b = Counter(temp)
    return b.most_common(1)


# data_path = 'ACDC_dataset/training/patient001/patient001_frame01.nii.gz'
# pred_path = 'ACDC_dataset/training/patient001/patient001_frame01_gt.nii.gz'
#
# metrics_acdc.main(data_path, data_path)
#
# data = nib.load(data_path)
# mask = nib.load(pred_path)
# print(data)
# # masker = NiftiMasker(mask_img=mask, standardize=True)
# plotting.plot_roi(mask, bg_img=data,
#                   cmap='Paired')
#
# plotting.show()






# plotting.plot_img(first_img)

# header = data.header
#
# plotting.plot_img(header)
# plotting.show()
# print(data)

# data = img.get_data()


"""
Shows all 30 images of first 4D image
"""
#
# for img in image.iter_img(data):
#     header = img.header
#     masker = NiftiMasker(mask_img=header, standardize=True)
#     plotting.plot_roi(masker, bg_img=img,
#                       cmap='Paired')
#     # plotting.plot_img(first_img)
#
#     plotting.show()
