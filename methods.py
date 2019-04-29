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
    """

    :param data: np array (979, 64, 64)
    :return:
    """

    for i in data:
        i = i*1.
        i = np.clip(i, 0, np.percentile(i, 99))

        i = i - np.amin(i)
        if np.amax(i) != 0:
            i = i / np.amax(i)
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


def load_images(data_dir, raw_image_path01, raw_image_path12, label_path01, label_path12):
    """

    :param data_dir, both paths to systole and diastole images with corresponding labels
    :return: array of paths to unlabeled images and array with path to corresponding labels
    """
    im_data = []
    labels = []
    for o in os.listdir(data_dir): # o: patientxxx
        # print('o: ', o)
        if not o.startswith('.') and os.path.isdir(os.path.join(data_dir, o)):

            for x in os.listdir(os.path.join(data_dir, o)): # x: patientXXX_frameXX.nii.gz

                if not x.startswith('.'):
                    # print('x: ' , data_dir + '/' + x)
                    # print('path: ', os.path.join(data_dir, o) + raw_image_path01)
                    # print(data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path01) or (
                    #         data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path12)
                    if (data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path01) or (
                            data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path12):
                        im_data.append(data_dir + '/' + o + '/' + x)

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
    show_slices([slice_0])


    plt.suptitle("Center slices for image")
    plt.show(block = block)


def find_center_of_mass(data):
    """

    :param data: array of images with shape (n_i, n_j, n_k)
    :return: list of arrays (of length n_k)  with the center of mass for each of these images (n_i, n_j) and list of empty labels
    """
    coms = []
    print(np.shape(data))
    empty_labels = []
    for s in range(0, len(data)):
        com = []
        for i in range(0, data[s].shape[2]):
            temp = np.array(sum(data[s][..., [i]]))
            bool_temp = temp != 0

            if sum(bool_temp) != 0:         #verify that label is non empty
                com.append(nd.measurements.center_of_mass(data[s][..., [i]][..., 0]))

            else: empty_labels.append((s, i))

        coms.append(com)

    print(np.shape(coms))
    return coms, empty_labels


def crop_images(cropper_size, center_of_masses, data):
    """

    :param cropper_size:
    :param center_of_masses:
    :param data:
    :return: returns np.array of cropped images
    """
    cropped_data = []
    for s in range(0, len(data)-1):
        temp = np.empty((data[s].shape[2], 2*cropper_size, 2*cropper_size))
        for i in range(0, data[s].shape[2]-1):

            center_i = int(center_of_masses[s][i][0])
            center_j = int(center_of_masses[s][i][1])

            if center_j - cropper_size > 0 and center_i - cropper_size > 0:

                # print('center_i - cropper_size', center_i - cropper_size)
                # print('center_j - cropper_size', center_j - cropper_size)
                # TODO for cropper_size = 64, center_j - cropper_size = -1 -> need to pad?
                temp[i] = data[s][..., i][center_i - cropper_size: center_i + cropper_size, center_j - cropper_size: center_j + cropper_size]

        cropped_data.append(temp)
    return cropped_data




def remove_empty_label_data(data, empty_labels):
    for i in empty_labels:

        data[i[0]] = np.delete(data[i[0]], i[1], 2)
    return data





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
