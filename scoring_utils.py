import os, torch, math
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
from matplotlib import gridspec




def evaluate(probas_, y_test, mask_test, n_subjects, n_x, n_y, n_z):
    """
    inputs need to be flattened
    :param probas_: y_pred
    :param y_test:
    :param mask_test: ones with same shape as y
    :param n_subjects:
    :param n_x: 64
    :param n_y: 64
    :param n_z: 1
    :return:
    """
    probas_ = np.squeeze(probas_)
    if probas_.shape != y_test.shape:
        print('PROBAS AND TEST IMAGE DO NOT HAVE THE SAME SHAPE', probas_.shape, y_test.shape)
    # Voxel-wise statistics
    # Compute ROC curve, area under the curve, f1, and accuracy
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:])
    roc_auc = auc(fpr, tpr)
    # get optimal cutOff
    threshold = cutoff_youdens_j(fpr, tpr, thresholds)
    print('Using threshold:', str(threshold))
    # threshold = 0.5 # threshold choosen to evaluate f1 and accuracy of model

    jaccard = jaccard_similarity_score(y_test, probas_[:] >= threshold)
    accuracy = accuracy_score(y_test, probas_[:] >= threshold)
    f1 = f1_score(y_test, probas_[:] >= threshold)

    # Image-wise statistics
    thresholded_volume_deltas = []
    unthresholded_volume_deltas = []
    image_wise_error_ratios = []
    image_wise_jaccards = []
    image_wise_hausdorff = []
    image_wise_modified_hausdorff = []
    image_wise_dice = []
    # figure for visual evaluation

    plt.switch_backend('agg')
    nrow = 2; ncol = n_subjects
    figure = plt.figure(figsize=(ncol+1, nrow+1))
    gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.0, hspace=0.0,
             top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
             left=0.5/(ncol+1), right=1-0.5/(ncol+1))

    vxl_index = 0

    for subj in range(n_subjects):
        subj_n_vxl = np.sum(mask_test[subj])
        subj_image_wise_probas = probas_[vxl_index : vxl_index + subj_n_vxl]
        subj_image_wise_y_test = y_test[vxl_index : vxl_index + subj_n_vxl]
        vxl_index += subj_n_vxl

        # Volume delta is defined as GT - predicted volume
        thresholded_volume_deltas.append(np.sum(subj_image_wise_y_test) - np.sum(subj_image_wise_probas >= threshold))
        unthresholded_volume_deltas.append(np.sum(subj_image_wise_y_test) - np.sum(subj_image_wise_probas))
        n_voxels = subj_image_wise_y_test.shape[0]

        # error ratio being defined as sum(FP + FN)/all
        image_wise_error_ratios.append(
            np.sum(abs(subj_image_wise_y_test - (subj_image_wise_probas >= threshold))) / n_voxels
        )
        image_wise_jaccards.append(jaccard_similarity_score(subj_image_wise_y_test, subj_image_wise_probas[:] >= threshold))
        image_wise_dice.append(dice(subj_image_wise_y_test, subj_image_wise_probas[:] >= threshold))

        # To calculate the hausdorff_distance, the image has to be rebuild as a 3D image
        subj_3D_probas = np.full(mask_test[subj].shape, 0, dtype = np.float64)
        subj_3D_probas[mask_test[subj]] = subj_image_wise_probas
        subj_3D_y_test = np.full(mask_test[subj].shape, 0)
        subj_3D_y_test[mask_test[subj]] = subj_image_wise_y_test

        # visual_compare(subj_3D_y_test, subj_3D_probas, n_subjects, subj, n_z, gs)

        # hsd, modified_hsd = hausdorff_distance(subj_3D_y_test, subj_3D_probas >= threshold, n_x, n_y, n_z)
        # image_wise_hausdorff.append(hsd)
        # image_wise_modified_hausdorff.append(modified_hsd)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'evaluation_threshold': threshold,
        'accuracy': accuracy,
        'jaccard': jaccard,
        'f1': f1,
        'roc_auc': roc_auc,
        'thresholded_volume_deltas': thresholded_volume_deltas,
        'unthresholded_volume_deltas': unthresholded_volume_deltas,
        'image_wise_error_ratios': image_wise_error_ratios,
        'image_wise_jaccards': image_wise_jaccards,
        # 'image_wise_hausdorff': image_wise_hausdorff,
        # 'image_wise_modified_hausdorff': image_wise_modified_hausdorff,
        'image_wise_dice': image_wise_dice,
        'figure': figure
        }

def cutoff_youdens_j(fpr, tpr, thresholds):
    j_scores = tpr-fpr # J = sensivity + specificity - 1
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    # im1 = np.asarray(im1).astype(np.bool)
    # im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    # intersection = np.logical_and(im1, im2)
    intersection = im1 * im2

    return 2. * intersection.sum() / im_sum

def hausdorff_distance(data1, data2, n_x, n_y, n_z):
    data1 = data1.reshape(n_x, n_y, n_z)
    data2 = data2.reshape(n_x, n_y, n_z)

    coordinates1 = np.array(np.where(data1 > 0)).transpose()
    coordinates2 = np.array(np.where(data2 > 0)).transpose()

    return directed_hausdorff(coordinates1, coordinates2)[0], ModHausdorffDist(coordinates1, coordinates2)[0]

def ModHausdorffDist(A,B):
    #This function computes the Modified Hausdorff Distance (MHD) which is
    #proven to function better than the directed HD as per Dubuisson et al.
    #in the following work:
    #
    #M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
    #matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
    #http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    #
    #The function computed the forward and reverse distances and outputs the
    #maximum/minimum of both.
    #Optionally, the function can return forward and reverse distance.
    #
    #Format for calling function:
    #
    #[MHD,FHD,RHD] = ModHausdorffDist(A,B);
    #
    #where
    #MHD = Modified Hausdorff Distance.
    #FHD = Forward Hausdorff Distance: minimum distance from all points of B
    #      to a point in A, averaged for all A
    #RHD = Reverse Hausdorff Distance: minimum distance from all points of A
    #      to a point in B, averaged for all B
    #A -> Point set 1, [row as observations, and col as dimensions]
    #B -> Point set 2, [row as observations, and col as dimensions]
    #
    #No. of samples of each point set may be different but the dimension of
    #the points must be the same.
    #
    #Edward DongBo Cui Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat,axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat,axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return(MHD, FHD, RHD)

# draw GT and test image on canvas
def visual_compare(GT, pred, n_images, i_image, n_z, gs):
    # center_z = (n_z - 1) // 2
    center_z = 1
    # plot GT image
    ax= plt.subplot(gs[0, i_image])
    plt.imshow(-GT[:, :, center_z].T)
    plt.gca().invert_yaxis()
    plt.set_cmap('Greys')
    plt.clim(-1, 0)
    plt.axis('off')

    # plot reconstructed image
    ax= plt.subplot(gs[1, i_image])
    plt.imshow(pred[:, :, center_z].T)
    plt.gca().invert_yaxis()
    plt.set_cmap('jet')
    plt.clim(0, 1)
    plt.axis('off')