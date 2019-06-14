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
            dice_threshold.append(dc(np.where(output[i][s] > threshold, 1, 0), y_test[i][s]))

    median_thrdice_score = np.median(dice_threshold)
    avr_thrdicesocre = np.average(dice_threshold)
    std = np.std(dice_threshold)

    return median_thrdice_score, std

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
    best_roc_idx = np.argmax([dict["median_ROC_AUC"] for dict in results])





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
    datapercs = [0.25, 0.5, 0.75, 1]
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

        for dict in results:
            if dict["model"] == whichmodel and dict["unet_layers"] == layers[counter] and dict["loss"] == whichloss:
                if dict["number_of_patients"] == datapercs[0]:
                    dices025.append(dict["median_thresholded_dice"])
                if dict["number_of_patients"] == datapercs[1]:
                    dices05.append(dict["median_thresholded_dice"])
                if dict["number_of_patients"] == datapercs[2]:
                    dices075.append(dict["median_thresholded_dice"])
                if dict["number_of_patients"] == datapercs[3]:
                    dices1.append(dict["median_thresholded_dice"])

                med_dice025 = round(np.median(dices025), 2)
                std_dice025 = round(np.std(dices025), 2)
                med_dice05 = round(np.median(dices05), 2)
                std_dice05 = round(np.std(dices05), 2)
                med_dice075 = round(np.mean(dices075), 2)
                std_dice075 = round(np.std(dices075), 2)
                med_dice1 = round(np.mean(dices1), 2)
                std_dice1 = round(np.std(dices1), 2)

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


def get_plot(whichdataset, whichmodel):

    basepath = whichdataset +'_results/'
    loss = 'binary_crossentropy'
    pers_percs = [0.25, 0.5]
    slice_percs = [0.25, 0.5, 0.75, 1]
    splits = [1,2,3,4]
    layers = [4]



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
                    if pers_perc == 0.25:
                        dices025.append(compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[0])
                        dices025_std.append(compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[1])
                    if pers_perc == 0.5:
                        dices05.append(compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[0])
                        dices05_std.append(compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[1])
                    if pers_perc == 0.75:
                        dices075.append(compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[0])
                        dices075_std.append(compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[1])
                    if pers_perc == 1:
                        dices1.append(compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[0])
                        dices1_std.append(compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[1])



        rounding_num = 2
        med_dice025 = round(np.median(dices025), rounding_num)
        std_dice025 = round(np.median(dices025_std),rounding_num)
        med_dice05 = round(np.median(dices05),rounding_num)
        std_dice05 = round(np.median(dices05_std),rounding_num)
        med_dice075 = round(np.mean(dices075),rounding_num)
        std_dice075 = round(np.median(dices075_std),rounding_num)
        med_dice1 = round(np.mean(dices1),rounding_num)
        std_dice1 = round(np.median(dices1_std),rounding_num)

        if layer == layers[0]:
            param_unet2_med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]
            param_unet2_std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]
        if layer == layers[1]:
            param_unet3_med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]
            param_unet3_std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]
        if layer == layers[2]:
            param_unet4_med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]
            param_unet4_std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]
        if layer == layers[3]:
            param_unet5_med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]
            param_unet5_std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]
        if layer == layers[4]:
            param_unet6_med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]
            param_unet6_std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]
        if layer == layers[5]:
            param_unet7_med_dices = [med_dice025, med_dice05, med_dice075, med_dice1]
            param_unet7_std_dices = [std_dice025, std_dice05, std_dice075, std_dice1]

    means = [param_unet2_med_dices,param_unet3_med_dices,param_unet4_med_dices,param_unet5_med_dices,param_unet6_med_dices,param_unet7_med_dices]
    stds = [param_unet2_std_dices,param_unet3_std_dices,param_unet4_std_dices,param_unet5_std_dices,param_unet6_std_dices,param_unet7_std_dices]
    fig, (ax) = plt.subplots(1)
    ind = np.arange(len(means))
    width = 0.2

    ax.bar(ind - width * 3 / 2, means[0], width, yerr=stds[0], label='1')
    ax.bar(ind - width * 1 / 2, means[1], width, yerr=stds[1], label='2')
    ax.bar(ind + width * 1 / 2, means[2], width, yerr=stds[2], label='3')
    ax.bar(ind + width * 3 / 2, means[3], width, yerr=stds[3], label='4')

    names = ['25%', '50%', '75%', '100%']
    ax.set_ylim(0, 1)
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    ax.set_xlabel('of total patients')
    ax.set_title('Dice score vs percentages of patients (' + str(4) + ' evaluations per bar)')
    ax.legend()
    fig.tight_layout()
    plt.show()



def plot_thrdice_vs_datapercs_for_model(whichdataset, whichmodel, layer):
    slice_percs = [1]
    basepath = whichdataset +'_results/'
    losses = ['binary_crossentropy']
    pers_percs = [0.25, 0.5]
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
                    if pers_perc == 0.25:
                        dices025.append(
                            compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[0])
                        dices025_std.append(
                            compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[1])
                    if pers_perc == 0.5:
                        dices05.append(compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[0])
                        dices05_std.append(
                            compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[1])
                    if pers_perc == 0.75:
                        dices075.append(
                            compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[0])
                        dices075_std.append(
                            compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[1])
                    if pers_perc == 1:
                        dices1.append(compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[0])
                        dices1_std.append(
                            compute_dice_score(whichmodel, loss, pers_perc, slice_perc, layer, split)[1])

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
    ax.set_title('Dice scores for '+ whichdataset + ' dataset against percentage of patients using '+ whichmodel + str(layer))
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
    plt.savefig(basepath+ whichmodel + str(layer))
    plt.show()

def save_plts():
    datasets = ['York']
    networks = ['param_unet']
    for dataset in datasets:
        for network in networks:
            if network == 'param_unet':
                layers = [4]
            else: layers = [1]
            for layer in layers:
                plot_thrdice_vs_datapercs_for_model(dataset, network, layer)


if __name__ == '__main__':
    save_plts()
    get_plot('York', 'param_unet')

something = 0