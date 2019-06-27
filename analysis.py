import os
from matplotlib import pyplot as plt
import numpy as np


def findpath(type, path):
    epochs = 0
    while not os.path.exists(path + '/' + str(epochs) + type):
        if epochs > 600:
            print("No data found")
            print(path)
            break
        else:
            epochs += 1
    if os.path.exists(path + '/' + str(epochs) + type): return True, path + '/' + str(epochs) + type
    epochs = 0
    while not os.path.exists(path + '/' + str(epochs) + 'max' + type):
        if epochs > 600:
            print("No data found")
            print(path)
            break
        else:
            epochs += 1
    if os.path.exists(path + '/' + str(epochs) + 'max' + type): return True, path + '/' + str(epochs) + 'max' + type
    return False, None


def get_prediction(path):
    predictions = []
    pred_bool, pred_path = findpath('epochs_mask_prediction.npy', path)
    if pred_bool:
        predictions = np.load(pred_path)
    else:
        print("DATA ERROR at", path)
    return np.array(predictions)


def read_value(path, score):
    if score == 'dice':
        index = path.find('_roundeddice')
        count = 0
        for i in range(1,5):
            if path[index - i].isdigit():
                count += 1
            else:
                value = float(path[index-count-2:index])
                break
    elif score == 'hd':
        index = path.find('_roundedhd')
        count = 0
        for i in range(1, 4):
            if path[index - i].isdigit():
                count += 1
            else:
                count2 = 0
                for j in range(1,4):
                    if path[index - count - 1 - j].isdigit():
                        count2 += 1
                    else:
                        value = float(path[index - count - count2 - 1:index])
                        break
    return value


def get_levels_plot(score, save, show, dataset):

    if dataset == "pgm":
        path = 'pgm_results/levels_final'
    if dataset == "nii":
        path = 'nii_results/levels_final'

    special = "100%_per_pat"
    params2 = ['2_levels', '3_levels', '4_levels', '5_levels']
    params = ['25%_total', '50%_total', '75%_total', '100%_total']

    data = [[[] for x in range(4)] for x in range(4)]

    for i in range(len(params)):
        for j in range(len(params2)):

            for root, subdirList, fileList in os.walk(path):
                for filename in fileList:
                    if score in filename and params[i] in root:  # check whether the file's DICOM
                        if params2[j] in root and special in root and "training_data" not in filename:
                            if score == "dice":
                                data[i][j].append(os.path.join(root, filename))
                            elif score == "hd":
                                if "nan" not in filename:
                                    data[i][j].append(os.path.join(root, filename))
    for i in range(len(data)):
        for j in range(len(data[i])):
            print("*********************************************************************************************")
            print(i, j, params[i], params2[j])
            for k in range(len(data[i][j])):
                print(read_value(data[i][j][k], score), data[i][j][k])
            values = [read_value(data[i][j][k], score) for k in range(len(data[i][j]))]
            print(round(np.median(values),3), round(np.std(values),3))

    medians = np.empty([4,4])
    stds = np.empty([4,4])
    evals = []

    for i in range(len(data)):
        for j in range(len(data[i])):
            values = [read_value(data[i][j][k], score) for k in range(len(data[i][j]))]
            medians[j,i] = np.median(values)
            stds[j,i] = np.std(values)
            evals.append(len(values))


    fig, (ax) = plt.subplots(1)
    ind = np.arange(len(medians))
    width = 0.2

    ax.bar(ind - width * 3 / 2, medians[0], width, yerr=stds[0],
           label=params2[0].replace('_', ' '))
    ax.bar(ind - width * 1 / 2, medians[1], width, yerr=stds[1],
           label=params2[1].replace('_', ' '))
    ax.bar(ind + width * 1 / 2, medians[2], width, yerr=stds[2],
           label=params2[2].replace('_', ' '))
    ax.bar(ind + width * 3 / 2, medians[3], width, yerr=stds[3],
           label=params2[3].replace('_', ' '))

    names = ['25%', '50%', '75%', '100%']

    if score == 'dice':
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1, 0.1))
    elif score == 'hd':
        ax.set_ylim(0, 60)
        ax.set_yticks(np.arange(0, 60, 5))

    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    ax.set_xlabel('of total patients')

    if score == "dice":
        ax.set_ylabel('Dice Score')
        ax.set_title(
            'Dice score vs percentages of patients (' + special.replace("_", " ") + ")")
    if score == "hd":
        ax.set_ylabel('Hausdorff Distance')
        ax.set_title(
            'Hausdorff Distance vs percentage of patients (' + special.replace("_", " ") + ")")

    print("evaluations:", evals)

    ax.set_axisbelow(True)
    plt.grid(axis='y')
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    if save:
        path = "plots/" + dataset + "-levels-" + score + ".png"
        plt.savefig(path)
        print("Saved Plot")
    plt.close()
    return None


def get_slices_plot(score, save, show, dataset, level):

    if dataset == "pgm":
        path = 'pgm_results/levels_final'
    if dataset == "nii":
        path = 'nii_results/levels_final'

    slice_percs = ['25%_per_pat', '50%_per_pat', '75%_per_pat', '100%_per_pat']
    patient_percs = ['25%_total', '50%_total', '75%_total', '100%_total']

    patients = [[] for x in range(4)]
    slices = [[] for x in range(4)]

    for i in range(len(patient_percs)):
        for root, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if score in filename and patient_percs[i] in root and "100%_per_pat" in root and level in root and "training_data" not in filename:  # check whether the file's DICOM
                    if score == "dice":
                        patients[i].append(os.path.join(root, filename))
                    elif score == "hd":
                        if "nan" not in filename:
                            patients[i].append(os.path.join(root, filename))

    for i in range(len(slice_percs)):
        for root, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if score in filename and slice_percs[i] in root and '100%_total_data' in root and level in root and "training_data" not in filename:  # check whether the file's DICOM
                    if score == "dice":
                        slices[i].append(os.path.join(root, filename))
                    elif score == "hd":
                        if "nan" not in filename:
                            slices[i].append(os.path.join(root, filename))

    for i in range(len(patients)):
        print("*********************************************************************************************")
        print(i, patient_percs[i], "100%_per_pat")
        for j in range(len(patients[i])):
            print(read_value(patients[i][j], score), patients[i][j])
        patient_values = [read_value(patients[i][j], score) for j in range(len(patients[i]))]
        print(round(np.median(patient_values),3), round(np.std(patient_values),3))

    for i in range(len(slices)):
        print("*********************************************************************************************")
        print(i, slice_percs[i], "100%_total_data")
        for j in range(len(slices[i])):
            print(read_value(slices[i][j], score), slices[i][j])
        slice_values = [read_value(slices[i][j], score) for j in range(len(slices[i]))]
        print(round(np.median(slice_values),3), round(np.std(slice_values),3))

    patient_medians = np.empty([4])
    patient_stds = np.empty([4])
    patient_evals = []

    slice_medians = np.empty([4])
    slice_stds = np.empty([4])
    slice_evals = []

    for i in range(len(slices)):
        values = [read_value(slices[i][j], score) for j in range(len(slices[i]))]
        slice_medians[i] = np.median(values)
        slice_stds[i] = np.std(values)
        slice_evals.append(len(values))

    for i in range(len(patients)):
        values = [read_value(patients[i][j], score) for j in range(len(patients[i]))]
        patient_medians[i] = np.median(values)
        patient_stds[i] = np.std(values)
        patient_evals.append(len(values))


    fig, (ax) = plt.subplots(1)
    ind = np.arange(len(patient_medians))
    width = 0.3

    ax.bar(ind - width / 2, patient_medians, width, yerr=patient_stds,
           label="% of patients")
    ax.bar(ind + width / 2, slice_medians, width, yerr=slice_stds,
           label="% of slices")

    names = ['25%', '50%', '75%', '100%']

    if score == 'dice':
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1, 0.1))
    elif score == 'hd':
        ax.set_ylim(0, 60)
        ax.set_yticks(np.arange(0, 60, 5))

    ax.set_xticks(ind)
    ax.set_xticklabels(names)

    if score == "dice":
        ax.set_ylabel('Dice Score')
        ax.set_title(
            'Dice score vs percentages of patients and slices (' + level.replace("_", " ") + ")")
    if score == "hd":
        ax.set_ylabel('Hausdorff Distance')
        ax.set_title(
            'Hausdorff Distance vs percentage of patients and slices (' + level.replace("_", " ") + ")")

    print("patient_evaluations:", patient_evals)
    print("slice_evaluations:", slice_evals)

    ax.set_axisbelow(True)
    plt.grid(axis='y')
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    if save:
        path = "plots/" + dataset + "-slices-" + score + ".png"
        plt.savefig(path)
        print("Saved Plot")
    plt.close()

    return None


def get_data_augm_plots(score, save, show, dataset):
    if dataset == "ACDC":
        path = 'ACDC_results_data_augm/new'
    if dataset == "York":
        path = 'York_results_data_augm/new'

    special = "100%_total_data-100%_per_pat"
    params = ['unaugm', '-r-', '-ws-', '-hs-', '-z-', '-hf-', '-vf-']
    param_names = ['unaugmented', 'rotation', 'width-shift', 'height-shift', 'zoom', 'hor-flip', 'vertical-flip']

    dice_data = [[] for x in range(7)]
    hd_data = [[] for x in range(7)]

    for i in range(len(params)):
        for root, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if params[i] in root:  # check whether the file's DICOM
                    if ("dice" in filename or "hd" in filename) and special in root and "training_data" not in filename:
                            dice_data[i].append(os.path.join(root, filename))
                            if "nan" not in filename:
                                hd_data[i].append(os.path.join(root, filename))

    def get_medians_stds_evals(data, score):
        medians = np.empty([7])
        stds = np.empty([7])
        evals = []
        for i in range(len(data)):
            print("*********************************************************************************************")
            print(i, params[i])
            for j in range(len(data[i])):
                print(read_value(data[i][j], score), data[i][j])
            values = [read_value(data[i][j], score) for j in range(len(data[i]))]
            print(round(np.median(values), 3), round(np.std(values), 3))
            medians[i] = np.median(values)
            stds[i] = np.std(values)
            evals.append(len(values))
        return medians, stds, evals

    dice_medians, dice_stds, dice_evals = get_medians_stds_evals(dice_data, "dice")
    hd_medians, hd_stds, hd_evals = get_medians_stds_evals(hd_data, "hd")

    fig, (ax1, ax2) = plt.subplots(2)
    ind = np.arange(len(dice_medians))
    width = 0.4

    ax1.bar(ind, dice_medians, width, yerr=dice_stds[0])
    ax2.bar(ind, hd_medians, width, yerr=hd_stds[0])

    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1, 0.1))
    ax2.set_ylim(0, 60)
    ax2.set_yticks(np.arange(0, 60, 5))

    ax1.set_xticks(ind)
    ax2.set_xticks(ind)
    ax1.set_xticklabels(param_names)
    ax2.set_xticklabels(param_names)
    ax1.set_xlabel('Data Augmentation Strategy')
    ax2.set_xlabel('Data Augmentation Strategy')

    ax1.set_ylabel('Dice Score')
    ax1.set_title(
            'Dice score vs Data Augmentation Strategies')
    ax2.set_ylabel('Hausdorff Distance')
    ax2.set_title(
            'Hausdorff Distance vs Data Augmentation Strategies')

    print("evaluations:", dice_evals)

    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.grid(axis='y')
    ax2.grid(axis='y')
    fig.tight_layout()
    if show:
        plt.show()
    if save:
        path = "plots/" + dataset + "-data_augm_strat.png"
        plt.savefig(path)
        print("Saved Plot")
    plt.close()
    return None


def get_plot():

    #dataset = "nii"
    dataset = "York"

    score = 'hd'
    #score = 'dice'

    #plot = "level"
    #plot = "slices"
    level = "4_levels"
    plot = "data_augm"

    show = True
    save = False

    if plot == "level":
        get_levels_plot(score, save, show, dataset)
    if plot == "slices":
        get_slices_plot(score, save, show, dataset, level)
    if plot == "data_augm":
        get_data_augm_plots(score, save, show, dataset)


if __name__=='__main__':

    get_plot()