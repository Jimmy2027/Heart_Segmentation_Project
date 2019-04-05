import acdc_data_loader as acdc


data_dir = 'ACDC_dataset/training'
raw_image_path01 = '_frame01.nii.gz'
raw_image_path12 = '_frame12.nii.gz'
label_path01 = '_frame01_gt.nii.gz'
label_path12 = '_frame12_gt.nii.gz'


images_paths, labels_paths = acdc.load_images(data_dir, raw_image_path01, raw_image_path12, label_path01, label_path12)
images_paths.sort()   #each label should be on the same index than its corresponding image
labels_paths.sort()

myocar_labels = acdc.remove_other_segmentations(labels_paths)  #array with all the myocar labels

img_data = acdc.load_data(images_paths)

print(myocar_labels.shape)
print(img_data.shape)