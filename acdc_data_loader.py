"""
TODO:

    - create unet with variable number of layers, check what goes wrong
    - york image preprocessing -> visualisation
    - tensorboard
    - center_of_mass for york dataset?

"""
import numpy as np
import methods
import matplotlib.pyplot as plt
import cv2 as cv



data_dir = 'ACDC_dataset/training'
raw_image_path01 = '_frame01.nii.gz'
raw_image_path12 = '_frame12.nii.gz'
label_path01 = '_frame01_gt.nii.gz'
label_path12 = '_frame12_gt.nii.gz'


images_paths, labels_paths = methods.load_images(data_dir, raw_image_path01, raw_image_path12, label_path01, label_path12)
images_paths.sort()   # each label should be on the same index than its corresponding image
labels_paths.sort()

myocar_labels = methods.remove_other_segmentations(labels_paths)  # array with all the myocar labels
img_data = methods.load_data(images_paths)  # array with all the heart images in slices
# all_labels = methods.load_data(labels_paths)
# methods.save_datavisualisation(img_data, all_labels, 'untouched_data/untouched_data', False)
# methods.save_datavisualisation2(img_data, all_labels, 'test/', False)




center_of_masses, empty_labels = methods.find_center_of_mass(myocar_labels)
empty_labels.sort(reverse=True)  # need to do this for remove_empty_labels



myocar_labels = methods.remove_empty_label_data(myocar_labels, empty_labels)       # need to remove empty labels from labels and image_data
img_data = methods.remove_empty_label_data(img_data, empty_labels)
image_counter = 0


# methods.save_datavisualisation2(img_data, myocar_labels, 'test/', False)

cropper_size = 64


resolution = 2 * cropper_size
cropped_img_data = methods.crop_images(cropper_size, center_of_masses, img_data)
cropped_myocar_labels = methods.crop_images(cropper_size, center_of_masses, myocar_labels)

for i in cropped_myocar_labels:
    for s in range(0, i.shape[0]):
        i[s] = i[s]//2




# methods.save_datavisualisation2(cropped_img_data, cropped_myocar_labels, 'data/after_cropping/', True)


unet_input = methods.data_normalization(cropped_img_data)


number_of_images = len(unet_input)




np.save('unet_input', unet_input)
np.save('unet_labels', cropped_myocar_labels)





# methods.save_datavisualisation(patientwise_preprocessed_images, patientwise_preprocessed_labels, 'after_normalization/preprocessed_data',False)
methods.save_datavisualisation2(unet_input, cropped_myocar_labels, 'network_input/', index_first=True, normalized=True)









