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
import imageio
import random
from sklearn import model_selection

def get_split(images, masks, split, seed):
    # split in form of (0.2,0.2)
    test_amount = int(images.shape[0]*split[0])
    val_amount = int(images.shape[0]*split[1])
    train_amount = images.shape[0] - test_amount - val_amount
    print("******************************************")
    print("TRAINING DATA: " + str(train_amount) + " patients")
    print("VALIDATION DATA: " + str(val_amount) + " patients")
    print("TEST DATA: " + str(test_amount) + " patients")
    print("******************************************")

    train_val_images, test_images, train_val_masks, test_masks = \
            model_selection.train_test_split(images, masks, test_size=test_amount, random_state=seed)

    train_images, val_images, train_masks, val_masks = \
            model_selection.train_test_split(train_val_images, train_val_masks, test_size=val_amount, random_state=seed)

    return train_images, train_masks, val_images, val_masks, test_images, test_masks

def get_splits(images, masks, splits, seed):
    # split in form of {1:(0.2, 0.2), 2:(0.3, 0.4)}
    # return: {split1: {train_images_split1: (100,96,96), train_masks_split1: ... }, split2: {}} for every split

    split_dicts = {}
    j = 0
    for split in splits.values():
        split_data = {}
        labels = ["train_images", "train_masks", "val_images", "val_masks",
                  "test_images", "test_masks"]
        train_images, train_masks, val_images, val_masks, test_images, test_masks = get_split(images, masks, split,
                                                                                              seed)
        data = [train_images, train_masks, val_images, val_masks, test_images, test_masks]
        for i in range(len(data)):
            split_data[labels[i]] = data[i]
        split_dicts["Split#" + str(j)] = split_data
        j += 1
    return split_dicts


def get_datasets(data, split_number):
    index = "Split#" + str(split_number)
    train_images = data[index].get("train_images")
    train_masks = data[index].get("train_masks")
    val_images = data[index].get("val_images")
    val_masks = data[index].get("val_masks")
    test_images = data[index].get("test_images")
    test_masks = data[index].get("test_masks")

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def getalldata(images, masks, data_percs, splits, seed):
    images_dict = {}
    masks_dict = {}
    split_dicts = {}
    for i in range(len(data_percs)):
        assert images.shape[0] == masks.shape[0]
        amount = int(images.shape[0] * data_percs[i])
        random.seed(seed)
        ind = random.sample(range(images.shape[0]), amount)
        images_dict[i] = images[ind]
        masks_dict[i] = masks[ind]

    for j in range(len(images_dict)):
        split_dicts[str(data_percs[j]) + "Perc"] = get_splits(images_dict[j], masks_dict[j], splits, seed)

    return split_dicts


def data_normalization(data):
    """

    :param
    :return:
    """

    for i in data:
        for j in range(0, i.shape[0]):
            print(i.shape)
            print(j)
            i[j] = i[j]*1.
            i[j] = np.clip(i[j], 0, np.percentile(i[j], 99))

            i[j] = i[j] - np.amin(i[j])
            if np.amax(i[j]) != 0:
                i[j] = i[j] / np.amax(i[j])
    return data



def remove_other_segmentations(labels):
    """
    loads labels and removes all other segmentations apart from the myocardium one

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
                    # print('path: ','/' os.path.join(data_dir, o) + raw_image_path01)
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

    :param data: array of images with shape (x, y, z)
    :return: list of arrays (of length z)  with the center of mass for each of these images (x, y) and list of empty labels (which patient, which slice)
    """
    coms = []
    # print(np.shape(data))
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

    # print(np.shape(coms))
    return coms, empty_labels

def york_find_center_of_mass(data):
    """

    :param data: array of images with shape (x, y, z)
    :return: list of arrays (of length z)  with the center of mass for each of these images (x, y) and list of empty labels (which patient, which slice)
    """
    coms = []
    # print(np.shape(data))
    empty_labels = []
    for s in range(0, len(data)):
        com = []
        for i in range(0, np.shape(data[s])[2]):
            temp = np.array(sum(data[s][..., [i]]))
            bool_temp = temp != 0

            if sum(bool_temp) != 0:         #verify that label is non empty
                com.append(nd.measurements.center_of_mass(data[s][..., [i]][..., 0]))

            else: empty_labels.append((s, i))

        coms.append(com)

    # print(np.shape(coms))
    return coms, empty_labels

def crop_images(cropper_size, center_of_masses, data):
    """

    :param cropper_size:
    :param center_of_masses: list of lists of length z with the com (x,y)
    :param data: array of images with shape (x, y, z)
    :return: returns np.array of cropped images (z, x, y)
    """
    cropped_data = []
    counter = 0

    for s in range(0, len(data)):
        temp = np.empty((data[s].shape[2], 2*cropper_size, 2*cropper_size))
        for i in range(0, data[s].shape[2]):

            center_i = int(center_of_masses[s][i][0])
            center_j = int(center_of_masses[s][i][1])

            if center_j - cropper_size > 0 and center_i - cropper_size > 0:

                # print('center_i - cropper_size', center_i - cropper_size)
                # print('center_j - cropper_size', center_j - cropper_size)

                temp[i] = data[s][..., i][center_i - cropper_size: center_i + cropper_size, center_j - cropper_size: center_j + cropper_size]

                # imageio.imwrite('visualisation/data/while_cropping/' + str(counter) + 'label' + '.png', temp[i,:,:])
                counter = counter + 1
            else:

                padded = np.pad(data[s][...,i],((64,64),(64,64)),'constant')

                temp[i] = padded[center_i + 64 - cropper_size: center_i + 64 + cropper_size, center_j + 64 - cropper_size: center_j + 64 + cropper_size]

        cropped_data.append(temp)


    return cropped_data




def remove_empty_label_data(data, empty_labels):
    """

    :param data:
    :param empty_labels: (which_person, which_slice)
    :return: data without the slices that have empty segemtnations
    """
    for i in empty_labels:

        data[i[0]] = np.delete(data[i[0]], i[1], 2)
    return data


def save_datavisualisation2(img_data, myocar_labels, save_folder, index_first = False, normalized = False):
    if index_first == True:
        for i in range(0, len(img_data)):
            img_data[i] = np.moveaxis(img_data[i], 0, -1)
            myocar_labels[i] = np.moveaxis(myocar_labels[i], 0, -1)

    counter = 0
    for i, j in zip(img_data[:], myocar_labels[:]):
        print(counter)
        print(i.shape)
        i_patch = i[:, :, 0]
        if normalized == True:
            i_patch = i_patch*255
        # np.squeeze(i_patch)

        j_patch = j[:, :, 0]
        # np.squeeze(j_patch)
        j_patch = j_patch * 255
        for slice in range(1, i.shape[2]):
            temp = i[:, :, slice]
            # np.squeeze(temp)
            if normalized == True:
                temp = temp * 255
            i_patch = np.hstack((i_patch, temp))


            temp = j[:, :, slice]
            # np.squeeze(temp)
            temp = temp * 255
            j_patch = np.hstack((j_patch, temp))

        image = np.vstack((i_patch, j_patch))

        print(image.shape)
        imageio.imwrite('visualisation/' + save_folder + '%d.png' % (counter,), image)
        counter = counter + 1



def save_datavisualisation3(img_data, myocar_labels, predicted_labels,median_dice_score, save_folder, index_first = False, normalized = False):
    img_data_temp = []
    myocar_labels_temp = []
    predicted_labels_temp = []
    if index_first == True:
        for i in range(0, len(img_data)):
            img_data_temp.append(np.moveaxis(img_data[i], 0, -1))
            myocar_labels_temp.append(np.moveaxis(myocar_labels[i], 0, -1))
            predicted_labels_temp.append(np.moveaxis(predicted_labels[i], 0, -1))
    counter = 0
    for i, j, k in zip(img_data_temp[:], myocar_labels_temp[:], predicted_labels_temp[:]):
        print(counter)
        print(i.shape)
        i_patch = i[:, :, 0]
        if normalized == True:
            i_patch = i_patch*255
        # np.squeeze(i_patch)

        j_patch = j[:, :, 0]
        # np.squeeze(j_patch)
        j_patch = j_patch * 255

        k_patch = k[:,:,0]
        k_patch = k_patch*255

        for slice in range(1, i.shape[2]):
            temp = i[:, :, slice]
            # np.squeeze(temp)
            if normalized == True:
                temp = temp * 255
            i_patch = np.hstack((i_patch, temp))


            temp = j[:, :, slice]
            # np.squeeze(temp)
            temp = temp * 255
            j_patch = np.hstack((j_patch, temp))

            temp = k[:,:,slice]
            temp = temp*255
            k_patch = np.hstack((k_patch, temp))

        image = np.vstack((i_patch, j_patch, k_patch))

        print(image.shape)
        imageio.imwrite(save_folder + median_dice_score + 'mds' + '%d.png' % (counter,), image)
        counter = counter + 1





def recreate(img_data, data):
    """
    rearranges the input data patientwise
    :param img_data:
    :return:
    """
    # data = np.load('unet_input.npy')
    # y_test = np.load('unet_labels.npy')
    data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
    # y_data = np.reshape(y_data, (y_data.shape[0], y_data.shape[1], y_data.shape[2]))
    patients_images = []
    # patients_labels = []
    counter = 0
    for person, t in enumerate(img_data):
        h = t.shape[2]
        x_temp = np.moveaxis(data[counter:counter+h, :, :], 0, -1)
        # y_temp = np.moveaxis(y_data[counter:counter+h, :, :], 0, -1)
        patients_images.append(x_temp)
        # patients_labels.append(y_temp)
        counter = counter + h + 1
    return patients_images




