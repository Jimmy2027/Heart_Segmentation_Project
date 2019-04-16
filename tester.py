import network_trainer as nt
import acdc_data_loader as acdc
import methods
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt


test_data_dir = 'ACDC_dataset/testing'
raw_image_path01 = '_frame01.nii.gz'
raw_image_path12 = '_frame12.nii.gz'
label_path01 = '_frame01_gt.nii.gz'
label_path12 = '_frame12_gt.nii.gz'


test_images_paths, test_labels_paths = methods.load_images(test_data_dir, raw_image_path01, raw_image_path12, label_path01, label_path12)
test_images_paths.sort()   # each label should be on the same index than its corresponding image
test_labels_paths.sort()

myocar_labels = methods.remove_other_segmentations(test_labels_paths)  # array with all the myocar labels
img_data = methods.load_data(test_images_paths)  # array with all the heart images
shape = img_data[0].shape


center_of_masses, empty_labels = methods.find_center_of_mass(myocar_labels)
empty_labels.sort(reverse=True)  # need to do this for remove_empty_labels



myocar_labels = methods.remove_empty_label_data(myocar_labels, empty_labels)       # need to remove empty labels from labels and image_data
img_data = methods.remove_empty_label_data(img_data, empty_labels)




cropper_size = 32
resolution = 2*cropper_size
cropped_img_data = methods.crop_images(cropper_size, center_of_masses, img_data)
cropped_myocar_labels = methods.crop_images(cropper_size, center_of_masses, myocar_labels)

number_of_images = 0
for s in cropped_img_data:
    for i in range(0, s.shape[0] - 1):
        number_of_images = number_of_images + 1


number_of_slices_per_image = []
unet_input = np.empty((number_of_images, resolution, resolution))
unet_labels = np.empty((number_of_images, resolution, resolution))
counter = 0
for s in cropped_img_data:
    number_of_slices_per_image.append(s.shape[0])
    for i in range(0, s.shape[0]-1):
        unet_input[counter, ...] = s[i, ...]

for s in cropped_myocar_labels:
    for i in range(0, s.shape[0]-1):
        unet_labels[counter, ...] = s[i, ...]


unet_input = methods.data_normalization(unet_input)
unet_labels = methods.data_normalization(unet_labels)

test_unet_input = unet_input.reshape((number_of_images, resolution, resolution, 1))
test_unet_labels = unet_labels.reshape((number_of_images, resolution, resolution, 1))

model = load_model('unet.hdf5')
results = model.predict(test_unet_input, verbose = 1)


np.save('results', results)
