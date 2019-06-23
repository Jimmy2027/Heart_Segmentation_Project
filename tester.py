from medpy.metric.binary import hd, dc
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import os
import glob
import scoring_utils as su
from textwrap import wrap
import matplotlib.cm as cm

basepath = 'York_results'
endfolder = False
modalites = os.listdir(basepath)

def read_dice_score(model, lossfunction, patients, layers, split):
    for folder in  os.listdir(os.path.join(basepath, model, lossfunction, str(patients)+'patients', str(layers)+'layers', str(split) + 'split')):
        if folder.endswith('results'):
            results = torch.load(os.path.join(basepath, model, lossfunction, str(patients)+'patients', str(layers)+'layers', str(split) + 'split', folder))
        if folder.endswith('y_pred.npy'):
            y_pred = np.load(os.path.join(basepath, model, lossfunction, str(patients)+'patients', str(layers)+'layers', str(split) + 'split', folder))
    return results, y_pred



def compute_dice_score(model, lossfunction, patients, slice_perc, layers, split):
    """
    Computes the median dice scor for a split and the std
    :param model:
    :param lossfunction:
    :param patients:
    :param slice_perc:
    :param layers:
    :param split:
    :return:
    """


    for folder in os.listdir(os.path.join(basepath, model, lossfunction, str(patients) + 'patients', str(slice_perc)+'slices',str(layers) + 'layers', str(split) + 'split')):
        if folder.endswith('y_pred.npy'):
            y_pred = np.load(os.path.join(basepath, model, lossfunction, str(patients) + 'patients', str(slice_perc)+'slices',str(layers) + 'layers', str(split) + 'split', folder))
        if folder.endswith('y_test.npy'):
            y_test = np.load(os.path.join(basepath, model, lossfunction, str(patients) + 'patients', str(slice_perc)+'slices',str(layers) + 'layers', str(split) + 'split', folder))

    output = []
    dice_threshold = []
    threshold = 0.5

    for i in range(len(y_test)):
        output.append(np.squeeze(y_pred[i]))
        for s in range(y_test[i].shape[0]):
            if np.max(np.where(output[i][s] > threshold, 1, 0)) != 0:
                dice_threshold.append(dc(np.where(output[i][s] > threshold, 1, 0), y_test[i][s]))

    if dice_threshold:
        median_thrdice_score = np.median(dice_threshold)
        avr_thrdicesocre = np.average(dice_threshold)
        std = np.std(dice_threshold)
        faulty = False
    else:
        faulty = True
        median_thrdice_score = 0
        std = 0

    return median_thrdice_score, std, faulty

def compute_hausd_dist(model, lossfunction, patients, slice_perc, layers, split):
    """
    Computes the median hausdorff distance for a split and the std
    :param model:
    :param lossfunction:
    :param patients:
    :param slice_perc:
    :param layers:
    :param split:
    :return:
    """


    for folder in os.listdir(os.path.join(basepath, model, lossfunction, str(patients) + 'patients', str(slice_perc)+'slices',str(layers) + 'layers', str(split) + 'split')):
        if folder.endswith('y_pred.npy'):
            y_pred = np.load(os.path.join(basepath, model, lossfunction, str(patients) + 'patients', str(slice_perc)+'slices',str(layers) + 'layers', str(split) + 'split', folder))
        if folder.endswith('y_test.npy'):
            y_test = np.load(os.path.join(basepath, model, lossfunction, str(patients) + 'patients', str(slice_perc)+'slices',str(layers) + 'layers', str(split) + 'split', folder))

    output = []
    thresholded_hausdorff = []
    threshold = 0.5

    for i in range(len(y_test)):
        output.append(np.squeeze(y_pred[i]))
        for s in range(y_test[i].shape[0]):
            if np.max(np.where(output[i][s] > threshold, 1, 0)) != 0:
                hausdorff = hd(np.where(output[i][s] > threshold, 1, 0), y_test[i][s])
                thresholded_hausdorff.append(hausdorff)

    if thresholded_hausdorff:
        median_hausd_dist = np.median(thresholded_hausdorff)
        print(median_hausd_dist)
        std = np.std(thresholded_hausdorff)
        faulty =False
    else:
        faulty = True
        median_hausd_dist = 0
        std = 0

    return median_hausd_dist, std, faulty

def find_results(path, results):
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            path1 = os.path.join(path, folder)
            find_results(path1, results)
        else:
            if not folder.startswith('.'):
                objects = []
                for filename in os.scandir(path):
                    objects.append(filename.name)
                for file in objects:
                    if file.endswith(("_results")):
                        result = torch.load(os.path.join(path, file))
                        results.append(result)
    return results




def print_best_scores(whitchdataset):
    basepath = whitchdataset + '_results/'
    results = []

    results = find_results(basepath, results)
    best_dice_idx = np.argmax([dict["median_thresholded_dice"] for dict in results])
    best_roc_idx = np.argmin([dict["median_thresholded_hausdorff"] for dict in results])





    print(' BEST MEDIAN DICE SCORE:', results[best_dice_idx]["median_thresholded_dice"], 'with', results[best_dice_idx]["number_of_patients"], 'number of patients, layers = ',results[best_dice_idx]['unet_layers'] , ', epochs =', results[best_dice_idx]["epochs"], ', model =', results[best_dice_idx]['model'], 'and split nr. ', results[best_dice_idx]['which_split'], 'using',results[best_dice_idx]['loss'], 'as loss' )
    print(' BEST MEDIAN ROC AUC:', results[best_roc_idx]["median_dice_score"], 'with', results[best_roc_idx]["number_of_patients"], 'number of patients, layers = ',results[best_roc_idx]['unet_layers'] , ', epochs =', results[best_roc_idx]["epochs"], ', model =', results[best_roc_idx]['model'],'and split nr. ', results[best_roc_idx]['which_split'])


    result, y_pred = read_dice_score(str(results[best_dice_idx]['model']), str(results[best_dice_idx]["loss"]), str(results[best_dice_idx]["number_of_patients"]), results[best_dice_idx]['unet_layers'], results[best_dice_idx]['which_split'])

    plt.hist(np.unique(y_pred[0]))
    plt.title('mds: ' + str(round(result['median_dice_score'], 4)) + '   ' + 'roc_auc: ' + str(round(result['median_ROC_AUC'], 4)) )
    plt.show()

    result,y_pred = read_dice_score(str(results[best_roc_idx]['model']), str(results[best_roc_idx]["loss"]), str(results[best_roc_idx]["number_of_patients"]), results[best_roc_idx]['unet_layers'],results[best_roc_idx]['which_split'])

    # results, y_pred = read_dice_score('param_unet', 'dice', 118, 5)
    # results, y_pred = read_dice_score('unet', 'binary_crossentropy', 118, 1)


    plt.hist(np.unique(y_pred[0]))
    plt.title('mds: ' + str(round(result['median_dice_score'], 4)) + '   ' + 'roc_auc: ' + str(round(result['median_ROC_AUC'], 4)) )
    plt.show()



def scatterplot_thrdice_vs_datapercs_vs_model(whichloss):
    results = []
    results = find_results(basepath, results)
    datapercs = [0.25, 0.5, 0.75, 1]
    dices_param_unet_2 = []
    dices_param_unet_3 = []
    dices_param_unet_4 = []
    dices_param_unet_5 = []
    dices_segnet = []
    dices = []
    datapercs = []
    datapercs_param_unet_2 = []
    datapercs_param_unet_3 = []
    datapercs_param_unet_4 = []
    datapercs_param_unet_5 = []
    datapercs_segnet = []

    for dict in results:
        if dict['model'] == 'param_unet' and dict['unet_layers'] == 2 and dict['loss'] == whichloss:
            dices_param_unet_2.append(dict["median_thresholded_dice"])
            datapercs_param_unet_2.append(dict["number_of_patients"])
        if dict['model'] == 'param_unet' and dict['unet_layers'] == 3 and dict['loss'] == whichloss:
            dices_param_unet_3.append(dict["median_thresholded_dice"])
            datapercs_param_unet_3.append(dict["number_of_patients"])
        if dict['model'] == 'param_unet' and dict['unet_layers'] == 4 and dict['loss'] == whichloss:
            dices_param_unet_4.append(dict["median_thresholded_dice"])
            datapercs_param_unet_4.append(dict["number_of_patients"])
        if dict['model'] == 'param_unet' and dict['unet_layers'] == 5 and dict['loss'] == whichloss:
            dices_param_unet_5.append(dict["median_thresholded_dice"])
            datapercs_param_unet_5.append(dict["number_of_patients"])
        if dict['model'] == 'segnet' and dict['loss'] == whichloss:
            dices_segnet.append(dict["median_thresholded_dice"])
            datapercs_segnet.append(dict["number_of_patients"])

    dices = [np.median(dices_param_unet_2), np.median(dices_param_unet_3), np.median(dices_param_unet_4), np.median(dices_param_unet_5), np.median(dices_segnet)]
    datapercs = [datapercs_param_unet_2, datapercs_param_unet_3, datapercs_param_unet_4, datapercs_param_unet_5, datapercs_segnet]
    colors = cm.rainbow(np.linspace(0, 1, len(dices)))

    plt.figure()  # colors for different networksd
    labels = ['param_unet_2', 'param_unet_3', 'param_unet_4', 'param_unet_5', 'segnet']
    for i in range(len(dices)):
        plt.xlabel('Data percentages')
        plt.ylabel('Dice scores')
        plt.scatter(datapercs[i], dices[i], c=colors[i], s=50, label=labels[i])
    plt.legend(loc=(0.67, 0.75))
    plt.title('Dice scores of different networks against the data percentages for ' + whichloss)
    plt.show()

def plot_thrdice_vs_datapercs(whichloss, which_dataset):
    basepath = which_dataset + '_results/'
    loss = whichloss
    pers_percs = [0.25, 0.5, 0.75, 1]
    slice_percs = [0.25, 0.5, 0.75, 1]
    splits = [1, 2, 3, 4]
    networks = ['param_unet', 'param_unet', 'param_unet', 'param_unet', 'segnetwork']
    layers = [2,3,4,5,1]
    rects = np.empty((2, len(networks)))
    med_dices_list = []
    std_dices_list = []
    for counter, whichmodel in enumerate(networks):
        results = []
        results = find_results(basepath, results)
        dices025 = []
        dices05 = []
        dices075 = []
        dices1 = []
        dices025_std = []
        dices05_std = []
        dices075_std = []
        dices1_std = []
        for layer in layers:
            for pers_perc in pers_percs:
                for slice_perc in slice_percs:
                    for split in splits:
                        dice_score, std_dice, faulty_dice = compute_dice_score(whichmodel, loss, pers_perc, slice_perc,layer,split)
                        if faulty_dice == False:
                            if pers_perc == 0.25:
                                dices025.append(
                                    dice_score)
                                dices025_std.append(
                                    std_dice)
                            if pers_perc == 0.5:
                                dices05.append(dice_score)
                                dices05_std.append(
                                    std_dice)
                            if pers_perc == 0.75:
                                dices075.append(
                                    dice_score)
                                dices075_std.append(
                                    std_dice)
                            if pers_perc == 1:
                                dices1.append(dice_score)
                                dices1_std.append(
                                    std_dice)

                rounding_num = 2
                med_dice025 = round(np.median(dices025), rounding_num)
                std_dice025 = round(np.median(dices025_std), rounding_num)
                med_dice05 = round(np.median(dices05), rounding_num)
                std_dice05 = round(np.median(dices05_std), rounding_num)
                med_dice075 = round(np.mean(dices075), rounding_num)
                std_dice075 = round(np.median(dices075_std), rounding_num)
                med_dice1 = round(np.mean(dices1), rounding_num)
                std_dice1 = round(np.median(dices1_std), rounding_num)

                med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]

                std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]

        med_dices_list.append(med_dices)
        std_dices_list.append(std_dices)

    ind = np.arange(len(med_dices))  # the x locations for the groups
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()

    rects0 = ax.bar(ind - width/2, med_dices_list[0], width, yerr=std_dices_list[0],
                        label=networks[0]+'2')
    rects1 = ax.bar(ind + width/2, med_dices_list[1], width, yerr=std_dices_list[1],
                    label=networks[1] + '3')
    rects2 = ax.bar(ind - width, med_dices_list[2], width, yerr=std_dices_list[2],
                    label=networks[2]+'4')
    rects3 = ax.bar(ind + width, med_dices_list[3], width, yerr=std_dices_list[3],
                    label=networks[3]+'5')
    rects4 = ax.bar(ind - 2*width, med_dices_list[4], width, yerr=std_dices_list[4],
                    label=networks[4])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Percentage of patients used')
    ax.set_ylabel('median Dice scores')
    ax.set_title('Dice scores by dataset against percentage of patients for binary_crossentropy')
    ax.set_xticks(ind)
    ax.set_xticklabels(('25%', '50%', '75%', '100%'))
    ax.legend()

    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom')

    autolabel(rects0)
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)


    fig.tight_layout()
    title = ax.set_title("\n".join(wrap(
        "Dice scores by dataset against percentage of patients for " + whichmodel + ' and ' + whichloss + ' as loss function ',
        60)))

    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    plt.show()


def scatter_pers_vs_slice(whichdataset, whichmodel, layer):
    basepath = whichdataset + '_results/'
    loss = 'binary_crossentropy'
    pers_percs = [0.25, 0.5, 0.75, 1]
    slice_percs = [0.25, 0.5, 0.75, 1]
    splits = [1, 2, 3, 4]
    layers = [1]

    dices025 = []
    dices05 = []
    dices075 = []
    dices1 = []
    dices025_std = []
    dices05_std = []
    dices075_std = []
    dices1_std = []

    for pers_perc in pers_percs:
        slice025 = []
        slice05 = []
        slice075 = []
        slice1 = []
        slice025_std = []
        slice05_std = []
        slice075_std = []
        slice1_std = []



        for slice_perc in slice_percs:
            for split in splits:
                dice_score, std_dice, faulty_dice = compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer,
                                                                       split)
                hausdorff_dist, std_haus, faulty_haus = compute_hausd_dist(whichmodel, loss, pers_perc, slice_perc,
                                                                           layer, split)
                assert faulty_dice != faulty_haus, 'faulty_dice != faulty_haus'
                if faulty_dice == False:
                    if pers_perc == 1:
                        if slice_perc == 0.25:
                            slice025.append(dice_score)
                            slice025_std.append(std_dice)
                        if slice_perc == 0.5:
                            slice05.append(dice_score)
                            slice05_std.append(std_dice)
                        if slice_perc == 0.75:
                            slice075.append(dice_score)
                            slice075_std.append(std_dice)
                        if slice_perc == 1:
                            slice1.append(dice_score)
                            slice1_std.append(std_dice)
                        if slice_perc == 1:
                            if pers_perc == 0.25:
                                dices025.append(
                                    dice_score)
                                dices025_std.append(
                                    std_dice)
                            if pers_perc == 0.5:
                                dices05.append(dice_score)
                                dices05_std.append(
                                    std_dice)
                            if pers_perc == 0.75:
                                dices075.append(
                                    dice_score)
                                dices075_std.append(
                                    std_dice)
                            if pers_perc == 1:
                                dices1.append(dice_score)
                                dices1_std.append(
                                    std_dice)
    rounding_num = 2
    med_dice025 = round(np.median(dices025), rounding_num)
    std_dice025 = round(np.median(dices025_std), rounding_num)
    med_dice05 = round(np.median(dices05), rounding_num)
    std_dice05 = round(np.median(dices05_std), rounding_num)
    med_dice075 = round(np.mean(dices075), rounding_num)
    std_dice075 = round(np.median(dices075_std), rounding_num)
    med_dice1 = round(np.mean(dices1), rounding_num)
    std_dice1 = round(np.median(dices1_std), rounding_num)

    bincross_med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]
    bincross_std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]

    # medians = [pers025, pers05, pers075, pers1]
    # stds = [std025, std05, std075, std1]


def plot_dice_vs_pers_slice(whichdataset, whichmodel, layer):
    """
    Plots the dice scores of one model for different person percentages and different slice percentages
    :param whichdataset:
    :param whichmodel:
    :return:
    """

    basepath = whichdataset +'_results/'
    loss = 'binary_crossentropy'
    pers_percs = [0.25, 0.5, 0.75, 1]
    slice_percs = [0.25, 0.5, 0.75, 1]
    splits = [1,2,3,4]
    layers = [1]


    pers025 = []
    pers05 = []
    pers075 = []
    pers1 = []
    std025 = []
    std05 = []
    std075 = []
    std1 = []

    for pers_perc in pers_percs:
        dices025 = []
        dices05 = []
        dices075 = []
        dices1 = []
        dices025_std = []
        dices05_std = []
        dices075_std = []
        dices1_std = []
        for slice_perc in slice_percs:
            for split in splits:
                dice_score, std_dice, faulty_dice = compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer,
                                                                       split)
                hausdorff_dist, std_haus, faulty_haus = compute_hausd_dist(whichmodel, loss, pers_perc, slice_perc,
                                                                           layer, split)
                assert faulty_dice != faulty_haus, 'faulty_dice != faulty_haus'
                if faulty_dice == False:
                    if slice_perc == 0.25:
                        dices025.append(dice_score)
                        dices025_std.append(std_dice)
                    if slice_perc == 0.5:
                        dices05.append(dice_score)
                        dices05_std.append(std_dice)
                    if slice_perc == 0.75:
                        dices075.append(dice_score)
                        dices075_std.append(std_dice)
                    if slice_perc == 1:
                        dices1.append(dice_score)
                        dices1_std.append(std_dice)
        if pers_perc == 0.25:
            pers025.append([np.median(dices025), np.median(dices05), np.median(dices075), np.median(dices1)])
            std025.append([np.median(dices025_std),np.median(dices05_std), np.median(dices075_std), np.median(dices1_std)])
        if pers_perc == 0.5:
            pers05.append([np.median(dices025), np.median(dices05), np.median(dices075), np.median(dices1)])
            std05.append([np.median(dices025_std),np.median(dices05_std), np.median(dices075_std), np.median(dices1_std)])
        if pers_perc == 0.75:
            pers075.append([np.median(dices025), np.median(dices05), np.median(dices075), np.median(dices1)])
            std075.append([np.median(dices025_std),np.median(dices05_std), np.median(dices075_std), np.median(dices1_std)])
        if pers_perc == 1:
            pers1.append([np.median(dices025), np.median(dices05), np.median(dices075), np.median(dices1)])
            std1.append([np.median(dices025_std),np.median(dices05_std), np.median(dices075_std), np.median(dices1_std)])



    means = [pers025,pers05,pers075,pers1]
    stds = [std025,std05,std075,std1]
    fig, (ax) = plt.subplots(1)
    ind = np.arange(len(means))
    width = 0.2

    rects1 = ax.bar(ind - width * 3 / 2, means[0][0], width, yerr=stds[0][0], label='25%')
    rects2 = ax.bar(ind - width * 1 / 2, means[1][0], width, yerr=stds[1][0], label='50%')
    rects3 = ax.bar(ind + width * 1 / 2, means[2][0], width, yerr=stds[2][0], label='75%')
    rects4 = ax.bar(ind + width * 3 / 2, means[3][0], width, yerr=stds[3][0], label='100%')

    names = ['25%', '50%', '75%', '100%']
    ax.set_ylim(0, 1)
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    ax.set_xlabel('of total patients')
    ax.set_title('Dice score vs percentages of patients and slices for ' + whichmodel + str(layer))
    ax.legend()

    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height,3)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom', rotation = 15)

    autolabel(rects1, "left")
    autolabel(rects2, "left")
    autolabel(rects3, "left")
    autolabel(rects4, "right")



    fig.tight_layout()
    plt.show()



def plot_thrdice_vs_datapercs_for_model(whichdataset, whichmodel, layer):           #TODO plot dice score and hausdorf distance
    slice_percs = [1]
    basepath = whichdataset +'_results/'
    losses = ['binary_crossentropy']
    pers_percs = [0.25, 0.5, 0.75, 1]
    splits = [1,2,3,4]
    for loss in losses:


        dices025 = []
        dices05 = []
        dices075 = []
        dices1 = []
        dices025_std = []
        dices05_std = []
        dices075_std = []
        dices1_std = []

        for pers_perc in pers_percs:
            for slice_perc in slice_percs:
                for split in splits:
                    dice_score, std_dice, faulty_dice = compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)
                    hausdorff_dist, std_haus, faulty_haus = compute_hausd_dist(whichmodel, loss, pers_perc, slice_perc, layer, split)
                    assert faulty_dice != faulty_haus, 'faulty_dice != faulty_haus'
                    if faulty_dice == False:
                        if pers_perc == 0.25:
                            dices025.append(
                                dice_score)
                            dices025_std.append(
                                std_dice)
                        if pers_perc == 0.5:
                            dices05.append(dice_score)
                            dices05_std.append(
                                std_dice)
                        if pers_perc == 0.75:
                            dices075.append(
                                dice_score)
                            dices075_std.append(
                                std_dice)
                        if pers_perc == 1:
                            dices1.append(dice_score)
                            dices1_std.append(
                                std_dice)

            rounding_num = 2
            med_dice025 = round(np.median(dices025), rounding_num)
            std_dice025 = round(np.median(dices025_std), rounding_num)
            med_dice05 = round(np.median(dices05), rounding_num)
            std_dice05 = round(np.median(dices05_std), rounding_num)
            med_dice075 = round(np.mean(dices075), rounding_num)
            std_dice075 = round(np.median(dices075_std), rounding_num)
            med_dice1 = round(np.mean(dices1), rounding_num)
            std_dice1 = round(np.median(dices1_std), rounding_num)

            if loss == losses[0]:
                bincross_med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]
                bincross_std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]
            # if loss == losses[1]:
            #     dice_med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]
            #     dice_std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]


    ind = np.arange(len(bincross_med_dices))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width / 2, bincross_med_dices, width, yerr=bincross_std_dices,
                    label='binary_cross')
    # rects2 = ax.bar(ind + width / 2, dice_med_dices, width, yerr=dice_std_dices,
    #                 label='Dice')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Percentage of patients used')
    ax.set_ylabel('median Dice scores')
    ax.set_title('Dice scores for ' + whichdataset + ' dataset against percentage of patients using ' + whichmodel + str(layer))
    ax.set_xticks(ind)
    ax.set_xticklabels(('25%', '50%', '75%', '100%'))
    ax.legend(loc = 2)

    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom')

    autolabel(rects1, "left")
    # autolabel(rects2, "right")

    fig.tight_layout()
    title = ax.set_title("\n".join(wrap('Dice scores for '+ whichdataset + ' dataset against percentage of patients using '+ whichmodel + str(layer), 60)))

    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    plt.savefig(basepath + whichmodel + str(layer))
    plt.show()



def plot_thrdice_andHausd_vs_datapercs_for_model(whichdataset, whichmodel, layer):
    slice_percs = [1]
    basepath = whichdataset +'_results/'
    loss= 'binary_crossentropy'
    scores = ['Dice', 'Hausd']
    pers_percs = [0.25, 0.5, 0.75, 1]
    splits = [1,2,3,4]
    for score in scores:


        dices025 = []
        dices05 = []
        dices075 = []
        dices1 = []
        dices025_std = []
        dices05_std = []
        dices075_std = []
        dices1_std = []

        hausd025 = []
        hausd05 = []
        hausd075 = []
        hausd1 = []
        hausd025_std = []
        hausd05_std = []
        hausd075_std = []
        hausd1_std = []

        for pers_perc in pers_percs:
            for slice_perc in slice_percs:
                for split in splits:
                    dice_score, std_dice, faulty_dice = compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)
                    hausdorff_dist, std_haus, faulty_haus = compute_hausd_dist(whichmodel, loss, pers_perc, slice_perc, layer, split)
                    assert faulty_dice == faulty_haus, 'faulty_dice != faulty_haus'
                    if faulty_dice == False:
                        if pers_perc == 0.25:
                            dices025.append(
                                dice_score)
                            dices025_std.append(
                                std_dice)
                            hausd025.append(
                                hausdorff_dist)
                            hausd025_std.append(
                                std_haus)

                        if pers_perc == 0.5:
                            dices05.append(dice_score)
                            dices05_std.append(
                                std_dice)
                            hausd05.append(
                                hausdorff_dist)
                            hausd05_std.append(
                                std_haus)
                        if pers_perc == 0.75:
                            dices075.append(
                                dice_score)
                            dices075_std.append(
                                std_dice)
                            hausd075.append(
                                hausdorff_dist)
                            hausd075_std.append(
                                std_haus)
                        if pers_perc == 1:
                            dices1.append(dice_score)
                            dices1_std.append(
                                std_dice)
                            hausd1.append(
                                hausdorff_dist)
                            hausd1_std.append(
                                std_haus)

            rounding_num = 3
            med_dice025 = round(np.median(dices025), rounding_num)
            std_dice025 = round(np.median(dices025_std), rounding_num)
            med_dice05 = round(np.median(dices05), rounding_num)
            std_dice05 = round(np.median(dices05_std), rounding_num)
            med_dice075 = round(np.mean(dices075), rounding_num)
            std_dice075 = round(np.median(dices075_std), rounding_num)
            med_dice1 = round(np.mean(dices1), rounding_num)
            std_dice1 = round(np.median(dices1_std), rounding_num)

            med_hausd025 = round(np.median(hausd025), rounding_num)
            std_hausd025 = round(np.median(hausd025_std), rounding_num)
            med_hausd05 = round(np.median(hausd05), rounding_num)
            std_hausd05 = round(np.median(hausd05_std), rounding_num)
            med_hausd075 = round(np.mean(hausd075), rounding_num)
            std_hausd075 = round(np.median(hausd075_std), rounding_num)
            med_hausd1 = round(np.mean(hausd1), rounding_num)
            std_hausd1 = round(np.median(hausd1_std), rounding_num)

            if score == scores[0]:
                bincross_med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]
                bincross_std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]
            if score == scores[1]:
                hausd_med_dices = [med_hausd025, med_hausd05, med_hausd075, med_hausd1]
                hausd_std_dices = [std_hausd025, std_hausd05, std_hausd075, std_hausd1]


    ind = np.arange(len(bincross_med_dices))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width / 2, bincross_med_dices, width, yerr=bincross_std_dices,
                    label='Dice')
    rects2 = ax.bar(ind + width / 2, hausd_med_dices, width, yerr=hausd_std_dices,
                    label='Hausdorff')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Percentage of patients used')
    ax.set_ylabel('median Dice scores')
    ax.set_title('Dice scores for ' + whichdataset + ' dataset against percentage of patients using ' + whichmodel + str(layer))
    ax.set_xticks(ind)
    ax.set_xticklabels(('25%', '50%', '75%', '100%'))
    ax.legend(loc = 2)

    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom')

    autolabel(rects1, "left")
    autolabel(rects2, "right")

    fig.tight_layout()
    title = ax.set_title("\n".join(wrap('Dice scores for '+ whichdataset + ' dataset against percentage of patients using '+ whichmodel + str(layer), 60)))

    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    plt.savefig(basepath + whichmodel + str(layer))
    plt.show()


def save_plts():
    datasets = ['York']
    networks = ['param_unet']
    for dataset in datasets:
        for network in networks:
            if network == 'param_unet':
                layers = [2,3,4,5]
            else: layers = [1]
            for layer in layers:
                # plot_thrdice_vs_datapercs_for_model(dataset, network, layer)
                plot_thrdice_andHausd_vs_datapercs_for_model(dataset, network, layer)


if __name__ == '__main__':
    # save_plts()
    # plot_thrdice_andHausd_vs_datapercs_for_model('York', 'param_unet', 4)
    # plot_dice_vs_pers_slice('York', 'param_unet', 4)
    # scatter_pers_vs_slice('York', 'param_unet', 4)
    plot_thrdice_vs_datapercs('binary_crossentropy', 'York')

something = 0