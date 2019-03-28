import scipy.io
import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib
import os
from nilearn import image
from nilearn.input_data import NiftiMasker
import metrics_acdc


"""
Data shape: (216, 256, 10, 30)
"""

data_dir = 'ACDC_dataset/training'
raw_image_path01 = '_frame01.nii.gz'
raw_image_path12 = '_frame12.nii.gz'
label_path01 = '_frame01_gt.nii.gz'
label_path12 = '_frame12_gt.nii.gz'



def load_images(data_dir):
    im_data = []
    labels = []
    for o in os.listdir(data_dir) :
        if not o.startswith('.'):
           for x in os.listdir(os.path.join(data_dir, o))  and not x.startswith('.'):
               raw_images = [x if (x == os.path.join(data_dir, o, raw_image_path01)) or (x == os.path.join(data_dir, o, raw_image_path12))]
               
               im_data.append(raw_images)

               labels = [x if
                             x == (os.path.join(data_dir, o, label_path01)) or x==(
                                 os.path.join(data_dir, o, label_path12)) and not x.startswith('.')]

               labels.append(raw_images)

    return im_data, labels


images, labels = load_images(data_dir)

for i in images:
    plotting.plot_img(i)



# data_path = 'ACDC_dataset/training/patient001/patient001_frame01.nii.gz'
# pred_path = 'ACDC_dataset/training/patient001/patient001_frame01_gt.nii.gz'
#
# metrics_acdc.main(data_path, data_path)
#
# data = nib.load(data_path)
# mask = nib.load(pred_path)
#
# # masker = NiftiMasker(mask_img=mask, standardize=True)
# plotting.plot_roi(mask, bg_img=data,
#                   cmap='Paired')
#
# plotting.show()
#





# plotting.plot_img(first_img)

# header = data.header
#
# plotting.plot_img(header)
# plotting.show()
# print(data)

# data = img.get_data()


"""
Shows all 30 images
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
