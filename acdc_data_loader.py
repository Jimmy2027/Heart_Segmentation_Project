"""
TODO:
    - normalize the data
    - need test set


"""
import numpy as np
import methods


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
resolution = 2*cropper_size
cropped_img_data = methods.crop_images(cropper_size, center_of_masses, img_data)
cropped_myocar_labels = methods.crop_images(cropper_size, center_of_masses, myocar_labels)

number_of_slices_per_image = []
unet_input = np.empty((979, resolution, resolution))
unet_labels = np.empty((979, resolution, resolution))
counter = 0
for s in cropped_img_data:
    number_of_slices_per_image.append(s.shape[0])
    for i in range(0,s.shape[0]-1):
        unet_input[counter,...] = s[i,...]

for s in cropped_myocar_labels:
    for i in range(0,s.shape[0]-1):
        unet_labels[counter, ...] = s[i, ...]
# for s in range(0, 10):
#     for i in range(0, cropped_img_data[s].shape[2]-1):
#         plt.imshow(cropped_myocar_labels[s][i])
#         plt.show()
#         plt.imshow(cropped_img_data[s][i])
#         plt.show()

unet_input = methods.data_normalization(unet_input)
unet_labels = methods.data_normalization(unet_labels)

unet_input = unet_input.reshape((979, resolution,resolution, 1))
unet_labels = unet_labels.reshape((979, resolution,resolution, 1))




np.save('unet_input', unet_input)
np.save('unet_labels', unet_labels)

# for i in range(0, 10):
#     methods.display(cropped_myocar_labels[i])
#     methods.display(cropped_img_data[i])

# nt.network_trainer(cropper_size, unet_input, unet_labels)

def get_cropper_size():
    return cropper_size