import acdc_data_loader as acdc
import nibabel as nib



"""
Data shape: (216, 256, 10, 30)
"""

data_dir = 'ACDC_dataset/training'
raw_image_path01 = '_frame01.nii.gz'
raw_image_path12 = '_frame12.nii.gz'
label_path01 = '_frame01_gt.nii.gz'
label_path12 = '_frame12_gt.nii.gz'


images_paths, labels_paths = acdc.load_images(data_dir, raw_image_path01, raw_image_path12, label_path01, label_path12)
images_paths.sort()   #each label should be on the same index than its correspoding image
labels_paths.sort()

myocar_labels = acdc.remove_other_segmentations(labels_paths)  #array with all the myocar labels


for i in range(0,10):
    acdc.display(myocar_labels[i])




#TODO 1. remove all other segmentations apart from the myocardium one  2. find and store middles of every label





