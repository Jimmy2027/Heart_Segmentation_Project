
import numpy as np
import random
import keras.backend as K
import tensorflow as tf
import imageio
import os
import keras.preprocessing as kp


def save_visualisation(img_data, myocar_labels, predicted_labels, rounded_labels, median_rounded_dice_score,
                       median_thresholded_hausdorff, prefix, save_folder):
    counter = 0
    img_data = np.expand_dims(img_data, 0)
    myocar_labels = np.expand_dims(myocar_labels, 0)
    predicted_labels = np.expand_dims(predicted_labels, 0)
    rounded_labels = np.expand_dims(rounded_labels, 0)
    for i, j, k, l in zip(img_data, myocar_labels, predicted_labels, rounded_labels):
        i_patch = i[0, :, :] * 255
        j_patch = j[0, :, :] * 255
        k_patch = k[0, :, :] * 255
        l_patch = l[0, :, :] * 255
        for slice in range(1, i.shape[0]):
            i_patch = np.hstack((i_patch, i[slice, :, :] * 255))
            j_patch = np.hstack((j_patch, j[slice, :, :] * 255))
            k_patch = np.hstack((k_patch, k[slice, :, :] * 255))
            l_patch = np.hstack((l_patch, l[slice, :, :] * 255))
        image = np.vstack((i_patch, j_patch, k_patch, l_patch))
        imageio.imwrite(save_folder + '%s%s_roundeddice_%s_roundedhd.png' % (
        prefix, median_rounded_dice_score, median_thresholded_hausdorff), image)
        counter = counter + 1
        # print("Done visualising at", save_folder, '%s_%s_roundeddice_%s_roundedhd.png' % (prefix, median_rounded_dice_score, median_thresholded_hausdorff))


def save_visualisation2(img_data, myocar_labels, median_rounded_dice_score, median_thresholded_hausdorff, save_folder):
    counter = 0
    img_data = np.expand_dims(img_data, 0)
    myocar_labels = np.expand_dims(myocar_labels, 0)
    for i, j in zip(img_data, myocar_labels):
        i_patch = i[0, :, :] * 255
        j_patch = j[0, :, :] * 255
        for slice in range(1, i.shape[0]):
            i_patch = np.hstack((i_patch, i[slice, :, :] * 255))
            j_patch = np.hstack((j_patch, j[slice, :, :] * 255))
        image = np.vstack((i_patch, j_patch))
        imageio.imwrite(save_folder + 'training_data_%s_roundeddice_%s_roundedhd.png' % (
        median_rounded_dice_score, median_thresholded_hausdorff), image)
        counter = counter + 1
        # print("Done visualising at", save_folder, 'training_data_%s_roundeddice_%s_roundedhd.png' % (median_rounded_dice_score, median_thresholded_hausdorff))


def get_patient_split(pats_amount, split):
    test_perc = split[0]
    val_perc = split[1]
    train_perc = 1 - test_perc - val_perc

    test_pat_amount = max(int(test_perc * pats_amount), 1)
    val_pat_amount = max(int(val_perc * pats_amount), 1)
    train_pat_amount = max(int(train_perc * pats_amount), 1)

    indices = list(range(pats_amount))

    train_inds = random.sample(indices, train_pat_amount)
    train_pats = []
    for index in train_inds:
        train_pats.append(index)
        indices.remove(index)

    test_inds = random.sample(indices, test_pat_amount)
    test_pats = []
    for index in test_inds:
        test_pats.append(index)
        indices.remove(index)

    val_inds = random.sample(indices, val_pat_amount)
    val_pats = []
    for index in val_inds:  # remaining patients go to val (remaining after rounding train_perc and test_perc to int)
        val_pats.append(index)
        indices.remove(index)

    pat_splits = [train_pats, test_pats, val_pats]

    train_diff = train_perc - len(train_pats) / pats_amount * 100
    test_diff = test_perc - len(test_pats) / pats_amount * 100
    val_diff = val_perc - len(val_pats) / pats_amount * 100
    diffs = [train_diff, test_diff, val_diff]

    if len(indices) > 0:
        pat_splits[diffs.index(max(diffs))].append(indices[0])
        diffs.remove(max(diffs))
    if len(indices) > 1:
        pat_splits[diffs.index(max(diffs))].append(indices[1])
        diffs.remove(max(diffs))
    if len(indices) > 2:
        pat_splits[diffs.index(max(diffs))].append(indices[2])

    return train_pats, test_pats, val_pats


def get_total_perc_pats(pats, perc):
    amount = max(int(perc * len(pats)), 1)
    if perc - amount / len(pats) >= 0.5 * 1 / len(pats):
        amount += 1
    perc_pats = random.sample(pats, amount)
    return perc_pats


def get_slice_perc_split(total_imgs, total_masks, pats, perc):
    images = []
    masks = []

    for patient in pats:

        total_slices = len(total_imgs[patient])
        amount = max(int(perc * total_slices), 1)

        if perc - amount / total_slices >= 0.5 * 1 / total_slices:
            amount += 1

        indices = random.sample(range(total_slices), amount)

        for index in indices:
            images.append(total_imgs[patient][index])
            masks.append(total_masks[patient][index])

    images = np.array(images, dtype=float)
    masks = np.array(masks, dtype=float)

    return images, masks


def getdicescore(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def dice_coeff(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true, -1) + K.sum(y_pred, -1) + smooth)


def dice_coeff_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


def weighted_cross_entropy(y_true, y_pred, beta=0.7):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

        return tf.reduce_mean(loss)

    return loss(y_true, y_pred)


'''
def collectimages(mylist):
    data = []
    for patient in range(len(mylist)):
        for image in range(len(mylist[patient])):
            data.append((mylist[patient])[image])
    return data
def getpatpercs(images, masks, patperc):
    new_imgs = []
    new_masks = []
    for pat in range(len(images)):
        temp_imgs = []
        temp_masks = []
        for index in range(int(patperc*len(images[pat]))):
            temp_imgs.append((images[pat])[index])
            temp_masks.append((masks[pat])[index])
        new_imgs.append(temp_imgs)
        new_masks.append(temp_masks)
    return new_imgs, new_masks
'''


def matthews_coeff(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return K.eval(numerator / (denominator + K.epsilon()))


def threshold(images, upper, lower):
    images = np.array(images)
    images[images >= upper] = 1
    images[images < lower] = 0
    return images


def add_count(path):
    count = 0
    for root, subdirList, fileList in os.walk(path):
        for file in fileList:
            count += 1
    return count


def get_all_data(dataset, split, patient_perc, slice_perc):
    if dataset == 'pgm':
        (data_images, data_masks) = pgm.get_data()
    if dataset == 'nii':
        (images, masks) = nii.get_nii_data()
        data_images = []
        data_masks = []
        for i in range(len(images)):
            data_images.append(np.expand_dims(images[i], -1))
        for i in range(len(masks)):
            data_masks.append(np.expand_dims(masks[i], -1))


    train_pats, test_pats, val_pats = get_patient_split(len(data_images), split)

    train_pats = get_total_perc_pats(train_pats, patient_perc)
    test_pats = get_total_perc_pats(test_pats, 1)
    val_pats = get_total_perc_pats(val_pats, 1)

    train_images, train_masks = get_slice_perc_split(data_images, data_masks, train_pats, slice_perc, False)
    test_images, test_masks = get_slice_perc_split(data_images, data_masks, test_pats, 1, True)
    val_images, val_masks = get_slice_perc_split(data_images, data_masks, val_pats, 1, False)

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def get_args_list(data_augm, single_param, rotation_range, width_shift_range, height_shift_range, zoom_range,
                  horizontal_flip, vertical_flip):
    data_gen_args_list = []
    if data_augm:
        data_gen_args = dict(rotation_range=rotation_range,
                             width_shift_range=width_shift_range,
                             height_shift_range=height_shift_range,
                             zoom_range=zoom_range,
                             horizontal_flip=horizontal_flip,
                             vertical_flip=vertical_flip)
        if single_param:
            for augm_param in data_gen_args.keys():
                empty_args = dict(rotation_range=0,
                                  width_shift_range=0,
                                  height_shift_range=0,
                                  zoom_range=0,
                                  horizontal_flip=0,
                                  vertical_flip=0)
                value = data_gen_args.get(augm_param)
                if value != 0:
                    new_empty_args = empty_args
                    new_empty_args[augm_param] = value
                    data_gen_args_list.append(new_empty_args)
        else:
            data_gen_args_list.append(data_gen_args)
    else:
        data_gen_args_list.append(1)
    return data_gen_args_list


def get_train_generator(data_gen_args, train_images, train_masks, val_images, val_masks, seed, single_param, batch_size,
                        dataset):


    rotation_range = data_gen_args.get("rotation_range")
    width_shift_range = data_gen_args.get("width_shift_range")
    height_shift_range = data_gen_args.get("height_shift_range")
    zoom_range = data_gen_args.get("zoom_range")
    horizontal_flip = data_gen_args.get("horizontal_flip")
    vertical_flip = data_gen_args.get("vertical_flip")

    image_datagen = kp.image.ImageDataGenerator(**data_gen_args)
    mask_datagen = kp.image.ImageDataGenerator(**data_gen_args)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    image_datagen.fit(train_images, augment=True, seed=seed)
    mask_datagen.fit(train_masks, augment=True, seed=seed)

    x_augm_save_dir = 'something'
    prefix = 'else'


    image_generator = image_datagen.flow(
        train_images,
        seed=seed,batch_size=batch_size)

    mask_generator = mask_datagen.flow(
        train_masks,
        seed=seed, batch_size=batch_size)

    train_generator = zip(image_generator, mask_generator)

    augm_count_before = 0

    return train_generator, x_augm_save_dir, prefix, augm_count_before


