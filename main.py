"""
TODO:
    - need to flatten the slices for every person?
    - find segmentation centers
    - need to
    - normalize the data


"""
import acdc_data_loader as acdc
from matplotlib import pyplot as plt
import numpy as np


data_dir = 'ACDC_dataset/training'
raw_image_path01 = '_frame01.nii.gz'
raw_image_path12 = '_frame12.nii.gz'
label_path01 = '_frame01_gt.nii.gz'
label_path12 = '_frame12_gt.nii.gz'


images_paths, labels_paths = acdc.load_images(data_dir, raw_image_path01, raw_image_path12, label_path01, label_path12)
images_paths.sort()   # each label should be on the same index than its corresponding image
labels_paths.sort()

myocar_labels = acdc.remove_other_segmentations(labels_paths)  # array with all the myocar labels
img_data = acdc.load_data(images_paths)  # array with all the heart images
shape = img_data[0].shape


center_of_masses, empty_labels = acdc.find_center_of_mass(myocar_labels)
empty_labels.sort(reverse=True)  # need to do this for remove_empty_labels



myocar_labels = acdc.remove_empty_label_data(myocar_labels, empty_labels)       # need to remove empty labels from labels and image_data
img_data = acdc.remove_empty_label_data(img_data, empty_labels)




cropper_size = 32
cropped_img_data = acdc.crop_images(cropper_size, center_of_masses, img_data)
cropped_myocar_labels = acdc.crop_images(cropper_size, center_of_masses, myocar_labels)


# for s in range(0, 10):
#     for i in range(0, cropped_img_data[s].shape[2]-1):
#         plt.imshow(cropped_myocar_labels[s][i])
#         plt.show()
#         plt.imshow(cropped_img_data[s][i])
#         plt.show()




for i in range(0, 10):
    acdc.display(cropped_myocar_labels[i])
    acdc.display(cropped_img_data[i])









