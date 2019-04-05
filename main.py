"""
TODO:
    - find segmentation centers
    - need to
    - normalize the data


"""


import acdc_data_loader as acdc
import nibabel as nib
import numpy as np
from nilearn import plotting


"""
Data shape: (216, 256, 10)
"""

data_dir = 'ACDC_dataset/training'
raw_image_path01 = '_frame01.nii.gz'
raw_image_path12 = '_frame12.nii.gz'
label_path01 = '_frame01_gt.nii.gz'
label_path12 = '_frame12_gt.nii.gz'


images_paths, labels_paths = acdc.load_images(data_dir, raw_image_path01, raw_image_path12, label_path01, label_path12)
images_paths.sort()   #each label should be on the same index than its corresponding image
labels_paths.sort()

myocar_labels = acdc.remove_other_segmentations(labels_paths)  #array with all the myocar labels

img_data = acdc.load_data(images_paths
                          )
for i in img_data:
    print('data shape: ', i.shape)

# for i in range(0,10):
#     acdc.display(myocar_labels[i])

# normalized_data = acdc.data_normalization(img_data)
label_centers = []
centers_i = []
centers_j = []
centers_k = []
for i in myocar_labels:
    print('label shape: ', i.shape)
    label_centers.append(acdc.find_middle(i))
    centers_i.append(acdc.find_middle(i)[0])
    centers_j.append(acdc.find_middle(i)[1])
    centers_k.append(acdc.find_middle(i)[2])

average_i = sum(centers_i)//len(centers_i)
average_j = sum(centers_j)//len(centers_j)
average_k = sum(centers_k)//len(centers_k)
average_center = [average_i, average_j, average_k]
print('average center:', average_center)
cropper_size = 50
print(myocar_labels[0][average_i - cropper_size: average_i + cropper_size, average_j - cropper_size:average_j + cropper_size, :].shape)


most_common_dataShape = acdc.find_most_common_shape2(img_data)[0][0]
mc_dataShape_occurences = acdc.find_most_common_shape2(img_data)[0][1]          #only 12 images have this shape


print(most_common_dataShape)
print(mc_dataShape_occurences)




cropped_img_data, cropped_myocar_labels = acdc.crop_images2(cropper_size, most_common_dataShape, mc_dataShape_occurences, average_i, average_j, average_k, img_data, myocar_labels)

for i in range(0, mc_dataShape_occurences):
    acdc.display(cropped_myocar_labels[i])
    acdc.display(cropped_img_data[i])









