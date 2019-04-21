"""
TODO:
    - normalize the data
    - need test set


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
img_data = methods.load_data(images_paths)  # array with all the heart images
shape = img_data[0].shape


center_of_masses, empty_labels = methods.find_center_of_mass(myocar_labels)
empty_labels.sort(reverse=True)  # need to do this for remove_empty_labels



myocar_labels = methods.remove_empty_label_data(myocar_labels, empty_labels)       # need to remove empty labels from labels and image_data
img_data = methods.remove_empty_label_data(img_data, empty_labels)




cropper_size = 32


resolution = 2 * cropper_size
cropped_img_data = methods.crop_images(cropper_size, center_of_masses, img_data)
cropped_myocar_labels = methods.crop_images(cropper_size, center_of_masses, myocar_labels)



unet_input = np.concatenate(cropped_img_data, axis = 0)
unet_labels = np.concatenate(cropped_myocar_labels, axis = 0)



unet_input = methods.data_normalization(unet_input)


number_of_images = len(unet_input)
print(len(unet_input))


for i in range(len(unet_input) - 1, 0, -1):
    if np.count_nonzero(unet_input[i]) == 0 or np.count_nonzero(unet_labels[i]) == 0:

        unet_input = np.delete(unet_input, i, 0)
        unet_labels = np.delete(unet_labels, i, 0)


print(len(unet_input))

unet_input = np.expand_dims(unet_input, -1)  #-1 for the last element
unet_labels = np.expand_dims(unet_labels, -1)




np.save('unet_input', unet_input)
np.save('unet_labels', unet_labels)




def get_resolution():
    return resolution


#temp something


""" 
plot 10 first image with their labels
"""
#
# for s in range(0, 10):
#     for i in range(0, cropped_img_data[s].shape[2]-1):
#         # plt.imshow(cv.addWeighted(cropped_img_data[s][i], alpha= 1, src2 = cropped_myocar_labels[s][i], beta = 1, gamma = 0), plt.cm.gray)
#         # plt.show()
#         plt.imshow(cropped_myocar_labels[s][i], plt.cm.gray)
#         plt.show()
#         plt.imshow(cropped_img_data[s][i], plt.cm.gray)
#         plt.show()


#TODO why are there still empty images?
