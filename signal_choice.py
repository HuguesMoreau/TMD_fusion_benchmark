"""
Author Hugues
"""
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

from sklearn.metrics import mutual_info_score

import torch

from param import fs, duration_window, duration_overlap, duration_segment, classes_names

from main_shl import create_dataloaders, initialize_network, duplicate_in
from visualizations.classes import colors as color_per_class


def remove_suffix(string):
    "Removes everything that come after the '_copy' substring in the string"
    if '_copy' in string: # cut everything after the '_copy'
        _copy_index = string.find('_copy')
        return string[:_copy_index]
    else:
        return string




def show_scores(model, dataloader):
    """
    Computes the scores associated to a whole dataset, and plots the scores
    distibution

    Parameters
    ----------
    model (LateFusionCNN object)
    dataloader (pytorch DataLoader object), created with a separte_collate
        function (the input X is a dict)

    Returns
    -------
    scores_dict: a dictionnary of 2d arrays (shape: (samples, n_classes)), containing
        signal names as keys
    std_dict: a dictionnary of floats, containing the standard deviations of
        the scores from each signal
    """

    scores_dict = {}
    std_dict = {}

    n_classes = len(classes_names)
    for i_signal, signal in enumerate(model.signals_list):
        this_scores_list = []

        for i_batch, (X, Y) in enumerate(dataloader):
            for x in X.values(): x.requires_grad = False

            if '_copy' in signal: # cut everything after the '_copy'
                _copy_index = signal.find('_copy')
                root_signal = signal[:_copy_index]
            else:
                root_signal = signal

            signal_network = model.networks[signal]
#            this_scores_list.append(signal_network(X[root_signal]))
#
#        this_scores_array = torch.cat(this_scores_list, dim=0).cpu().detach().numpy()

            this_scores_list.append(signal_network(X[root_signal]).cpu().detach().numpy())

        this_scores_array = np.concatenate(this_scores_list, axis=0)
        scores_dict[signal] = this_scores_array

        """bins = np.linspace(this_scores_array.min(), this_scores_array.max(), 40+1) # shape: (41,)
        mean_bin = (bins[1:] + bins[:-1])/2    # shape: (40,)
        base = np.zeros((bins.shape[0]-1,))    # shape: (40,)
        width = bins[1] - bins[0]             # float
        plt.figure()
        for i_class in range(n_classes):
            this_class_hist = np.histogram(this_scores_array[:,i_class], bins=bins)[0]  # shape: (40,)
            plt.bar(mean_bin, this_class_hist, width, bottom=base)
            base += this_class_hist
        plt.legend(classes_names)
        plt.title(signal)
        plt.show()"""

        std_dict[signal] = this_scores_array.std()

    return scores_dict, std_dict







#%%
def create_train_record(split, data_type, fusion_type, signals_list,
                     log_power="missing", out_size="missing", interpolation="missing",
                     comp_preprocess_first=False, n_repetitions=1, maxepochs=50,
                     proportion_separate="missing"):
    """
    Similar to main_shl.create_and_train, but returns the weights along with
    the F1 scores. Only the modes with weights are accepted


    Parameters
    ----------
    (same as main_shl.create_and_train)
    - log_power, out_size, and interpolation are only mandatory when
        data_type == "spectrogram", and can be left ignored otherwise
    - comp_preprocess_first is only mandatory for the late fusions ('probas',
        'scores', and their weighted vatiants), and can be left ignored otherwise
    - comp_preprocess_first is True by default
    - n_repetitions (int) defines how many networks are trained using the same
        protocol, in order to average the results


    Returns
    -------
    avg_train_F1, std_train_F1, avg_val_F1, std_val_F1  (tuple of floats between 0 and 1)
    weights_list: a dict of lists, each list containing floats (keys are signal names)

    """

    if data_type == "spectrogram":
        missing_arguments = []

        if log_power     == "missing": missing_arguments.append("log_power")
        if out_size      == "missing": missing_arguments.append("out_size")
        if interpolation == "missing": missing_arguments.append("interpolation")

        if len(missing_arguments) >0:
            raise TypeError("Missing arguments: {}".format(",".join(missing_arguments)))

        if interpolation == 'none':
            n_points_frequency = int(duration_window*fs/2) +1
            n_points_temporal  = int((duration_segment-duration_window)/(duration_window-duration_overlap)) +1
            out_size = (n_points_frequency, n_points_temporal)

    if fusion_type in ["weighted_probas", "weighted_scores"] and  proportion_separate=="missing":
        raise TypeError("Missing argument: proportion_separate")


    if fusion_type not in ["weighted_probas", "weighted_scores", "GBlend", "bottleneck", "attention", "selective_fusion"]:
        raise TypeError(f"incorrect fusion_type argument: '{fusion_type}'")

    #  -----------------------  preprocessing  -------------------------
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(split, data_type, fusion_type,
                                       signals_list, log_power, out_size, interpolation, comp_preprocess_first, use_test=False)


    #  ---------------------------  Network training -----------------------------
    list_train_F1, list_val_F1 = [], []
    weight_lists = {}



    for i_repetition in range(n_repetitions):
        print(i_repetition, 'cuda.memory_allocated = %.2f Go' % (torch.cuda.memory_allocated()/10**9))
        network = initialize_network(data_type, fusion_type, signals_list, out_size, L2C_beta=-1, L2C_delta=-1, decorr_loss_weight=-1)


        if fusion_type in ["weighted_probas", "weighted_scores"]:
            _, _, train_f1, val_f1 = network.train_process(train_dataloader, val_dataloader,
                                                           maxepochs=maxepochs, proportion_separate=proportion_separate)
        else:
            _, _, train_f1, val_f1 = network.train_process(train_dataloader, val_dataloader, maxepochs=maxepochs)
        list_train_F1.append(train_f1)
        list_val_F1.append(val_f1)


        if duplicate_in(signals_list):
            # create a new list, with '_copy_', and an index added to the signal names
            copy_signals_list = [signal+f'_copy{i}' for (i, signal) in enumerate(signals_list)]
        else:
            copy_signals_list = signals_list

        model_weights = network.get_weights(copy_signals_list, val_dataloader)

        if fusion_type == "weighted_scores" and len(copy_signals_list) < 5 :    # do not plot a billion windows
            if i_repetition == 0: _, std_dict = show_scores(network, val_dataloader)

            model_weights = {signal:std_dict[signal]*model_weights[signal] for signal in model_weights}

        for signal in copy_signals_list:
            if signal not in weight_lists: weight_lists[signal] = []
            if fusion_type in ["selective_fusion", "attention"]:   # model_weights[signal] is an array of floats
                weight_lists[signal] += list(model_weights[signal])
            else : # model_weights[signal] is a float
                weight_lists[signal].append(model_weights[signal])

    return np.mean(list_train_F1), np.std(list_train_F1), np.mean(list_val_F1), np.std(list_val_F1), weight_lists




 #%%

if __name__ == "__main__":

    complete_signals_list = ['Acc_x',  'Acc_y',  'Acc_z',  'Acc_norm',
                    'Gra_x',  'Gra_y',  'Gra_z',  'Gra_norm',
                    'LAcc_x', 'LAcc_y', 'LAcc_z', 'LAcc_norm',
                    'Gyr_x',  'Gyr_y',  'Gyr_z',  'Gyr_norm',
                    'Mag_x',  'Mag_y',  'Mag_z',  'Mag_norm',
                    'Ori_x',  'Ori_y',  'Ori_z',  'Ori_w', 'Ori_norm',
                    'Pressure']


    possible_signals_lists = [["Acc_norm", "Gyr_y"],
                              ["Acc_norm", "Ori_norm"],
                              ["Acc_norm", "Acc_norm","Acc_norm","Acc_norm"]]
#                              ["Acc_norm", "Gyr_y","Mag_norm", "Ori_w"]]


    possible_fusion = ["attention", "selective_fusion", "weighted_scores", "weighted_probas", "GBlend", "bottleneck"]







    signals_lists_as_strings = [','.join(s_list) for s_list in possible_signals_lists]
    F1_scores_dict = {}
    weights = {(','.join(s_list)):
                    {fusion:
                        {}  # will contain {signals:list of weights}
                    for fusion in possible_fusion}
                for s_list in possible_signals_lists}

    for i_fusion, fusion in enumerate(possible_fusion):
        for i_signal, signals_list in enumerate(possible_signals_lists):

            if signals_list == complete_signals_list: # we will not have enough room in GPU memory for everything
                comp_preprocess_first = False
            else:
                comp_preprocess_first = True

            results = create_train_record(split="balanced", data_type="spectrogram",
                                   fusion_type=fusion, signals_list=signals_list,
                                   log_power=True, out_size=(48, 48),
                                   comp_preprocess_first=comp_preprocess_first,
                                   n_repetitions=5, proportion_separate=0.5,
                                   interpolation="log", maxepochs=50)
            _, _, mean_val_F1, std_val_F1, weight_lists = results

            key = str(signals_list) + "|" + fusion
            F1_scores_dict[key] = (mean_val_F1, std_val_F1)


            for signal in weight_lists.keys():
                if signal not in weights[signals_lists_as_strings[i_signal]][fusion]:
                    weights[signals_lists_as_strings[i_signal]][fusion][signal] = []
                weights[signals_lists_as_strings[i_signal]][fusion][signal] += weight_lists[signal]




import pickle
with open('signal_weights.pickle', 'wb') as f:
    pickle.dump(weights, f)



