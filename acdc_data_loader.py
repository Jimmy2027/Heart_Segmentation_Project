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
        img_data[img_data == 1 ] = 0
        img_data[img_data == 3 ] = 0
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
        print('o: ', o)
        if not o.startswith('.') and os.path.isdir(os.path.join(data_dir, o)):

            for x in os.listdir(os.path.join(data_dir, o)): #x: patientXXX_frameXX.nii.gz

                if not x.startswith('.'):
                    print('x: ' , data_dir + '/' + x)
                    print('path: ', os.path.join(data_dir, o) + raw_image_path01)
                    print(data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path01) or (
                            data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path12)
                    if (data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path01) or (
                            data_dir + '/' + x == os.path.join(data_dir, o) + raw_image_path12):
                        im_data.append(data_dir + '/' + o + '/'+ x)

                    if data_dir + '/' + x == (os.path.join(data_dir, o) + label_path01) or data_dir + '/' + x == (
                            os.path.join(data_dir, o) + label_path12):

                        labels.append(data_dir + '/'+ o+ '/' + x)

    return im_data, labels

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
    print('Image center: ', center_i, center_j, center_k)
    center_vox_value = img_data[center_i, center_j, center_k]
    print('Image center value: ', center_vox_value)

    slice_0 = img_data[center_i, :, :]
    slice_1 = img_data[:, center_j, :]
    slice_2 = img_data[:, :, center_k]

    show_slices([slice_0, slice_1, slice_2])

    plt.suptitle("Center slices for image")
    plt.show(block = block)









def plot_img_with_label(labels, images):
    """
    Plots 50 images with labels
    :param labels:
    :param images:
    """
    for i in range(1, 50):
        plotting.plot_roi(labels[i], bg_img=images[i], cmap='Paired')
        # plotting.plot_img(i)
        plotting.show()



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
