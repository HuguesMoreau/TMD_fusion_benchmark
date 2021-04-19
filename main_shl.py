"""
Author Hugues
"""


import torch
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

from param import fs, duration_window, duration_overlap, spectro_batch_size, duration_segment

from param import device
from preprocess import Datasets, transforms, fusion

from architectures import basic_CNN, late_fusion
from architectures.biblio_GB import GBlend_CNN
from architectures.biblio_learn2combine import L2C_CNN
from architectures.bottleneck_convolution import bottleneckCNN
from architectures.attention import AttentionCNN
from architectures.dissimilar_nets import DecorrelatedNet


def duplicate_in(l):
    """Returns True if there is at least one duplicated element in the list of
    strings l, and False otherwise"""
    l.sort()
    for i in range(len(l)-1):
        if l[i] == l[i+1]: return True
    return False

if __name__ == '__main__':
    assert (duplicate_in(['Acc_norm', 'Acc_x', 'Mag_z']) == False)
    assert (duplicate_in(['Acc_norm', 'Acc_x', 'Acc_norm']) == True)



def remove_duplicates(l):
    "Create a new list that contains the same elements as l, without duplicates"
    result = []
    for element in l:
        if element not in result:
            result.append(element)
    return result


if __name__ == '__main__':
    assert (remove_duplicates(['Acc_norm', 'Acc_x', 'Mag_z'])    == ['Acc_norm', 'Acc_x', 'Mag_z'] )
    assert (remove_duplicates(['Acc_norm', 'Acc_x', 'Acc_norm']) == ['Acc_norm', 'Acc_x'])


#%%
def create_dataloaders(split, data_type, fusion_type, signals_list,
                     log_power="missing", out_size="missing", interpolation="missing",
                     comp_preprocess_first=True, use_test=False):
    """
    generate the training, validation, and test sets with the given parameters,
    and returns the corresponding dataloaders

    Parameters
    ----------
    see above for inut type and constraints
    - log_power, out_size, and interpolation are only mandatory when
        data_type == "spectrogram", and can be left ignored otherwise
    - comp_preprocess_first is False by default
    - use_test (bool): if False, do not generate test dataloader. Default: False


    Returns
    -------
    train_dataloader, val_dataloader, test_dataloader
        tuple of torch.utils.data.DataLoader objects
        if use_test == False, test_dataloader is replaced with an empty list.

    """
    print("create_dataloaders", signals_list)

    if data_type in ["temporal", "FFT"]:
        if data_type == "temporal":
            transform_fn = transforms.TemporalTransform(remove_duplicates(signals_list))

        else :   #data_type == "FFT":
            transform_fn = transforms.FFTTransform(remove_duplicates(signals_list))

    elif data_type == "spectrogram":
        transform_fn = transforms.SpectrogramTransform(remove_duplicates(signals_list), fs, duration_window, duration_overlap, spectro_batch_size, interpolation, log_power, out_size)



    if fusion_type in ["time", "freq", "depth"]:
        collate_fn = fusion.ConcatCollate(fusion_type, list_signals=signals_list)
    elif fusion_type in ["probas", "scores", "weighted_probas", "weighted_scores",
                         "GBlend", "learn2combine", "decorrelated_classic", "decorrelated_deep"]:
        collate_fn = fusion.separate_sensors_collate
    elif fusion_type in ["features", "bottleneck", "attention", "selective_fusion"]:
        collate_fn = fusion.ConcatCollate("depth", list_signals=signals_list)
            # a 'depth' collate can be used for feature concatenation (intermediate fusion)
            # thanks to the 'group' argument of convolutional layers
            # see the documentation of basic_CNN for complete explanations


    train_dataset = Datasets.SignalsDataSet(mode='train', split=split, comp_preprocess_first=comp_preprocess_first, transform=transform_fn)
    val_dataset =   Datasets.SignalsDataSet(mode='val',   split=split, comp_preprocess_first=comp_preprocess_first, transform=transform_fn)

    if use_test:
        test_dataset = Datasets.SignalsDataSet(mode='test',  split=split, comp_preprocess_first=comp_preprocess_first, transform=transform_fn)


    batch_size = 64 if fusion_type != 'decorrelated_deep' else 512 # we need full-rank correlation matrices estimation for deep CCA
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, collate_fn=collate_fn, shuffle=use_test)
    if use_test:
        test_dataloader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    else:
        test_dataloader = []

    return  train_dataloader, val_dataloader, test_dataloader



#%%
def initialize_network(data_type, fusion_type, signals_list, out_size,
                       L2C_beta, L2C_delta, decorr_loss_weight):
    """
    initialize a neural network to train.

    Parameters
    ----------
    see above for inut type and constraints
    - out_size is ignored if data_type is not "spectrogram"

    Returns: a neural net object, with methods train_process and test
    """
    print(data_type, fusion_type)
    n_signals = len(signals_list)


    print("initialize_network", signals_list)


    if duplicate_in(signals_list):
        # create a new list, with '_copy_', and an index added to the signal names
        signals_list = [signal+f'_copy{i}' for (i, signal) in enumerate(signals_list)]


    if data_type in ["temporal", "FFT"]:
        if fusion_type == "time":
            input_shape = (1,         duration_segment*fs*n_signals)
        elif fusion_type in ["depth", "bottleneck"]:
            input_shape = (n_signals, duration_segment*fs)
        elif fusion_type == "features":
            input_shape = (1,         duration_segment*fs)
        elif fusion_type in ["probas", "scores", "weighted_probas", "weighted_scores",
                             "GBlend", "learn2combine", "selective_fusion", "attention",
                             "decorrelated_classic", "decorrelated_deep"]:
            input_shape = (1,         duration_segment*fs)


    elif data_type == "spectrogram":
        if fusion_type == "time":
            input_shape = (1,         out_size[0],           out_size[1]*n_signals)
        elif fusion_type == "freq":
            input_shape = (1,         out_size[0]*n_signals, out_size[1])
        elif fusion_type in ["depth", "bottleneck"]:
            input_shape = (n_signals, out_size[0],           out_size[1])
        elif fusion_type == "features":
            input_shape = (1,         out_size[0],           out_size[1])
        elif fusion_type in ["probas", "scores", "weighted_probas", "weighted_scores",
                             "GBlend", "learn2combine", "selective_fusion", "attention",
                             "decorrelated_classic", "decorrelated_deep"]:
            input_shape = (1,         out_size[0],           out_size[1])

        else :
            raise NotImplementedError("fusion_type: %s not implemented " % fusion_type)


    if fusion_type in ["time", "freq", "depth", "features"]:
        n_branches = n_signals if fusion_type == "features" else 1
        fusion_layer = "conv2" if fusion_type == "features" else "start"

        net = basic_CNN.CNN(input_shape, n_branches=n_branches, fusion=fusion_layer)


    elif fusion_type == "bottleneck":
        net = bottleneckCNN(input_shape)


    elif fusion_type in ["probas", "scores", "weighted_probas", "weighted_scores"]:
        use_weights = "weighted_" in fusion_type

        if use_weights :
            fusion_type = fusion_type[9:]
                # remove the "weighted_"

        net = late_fusion.LateFusionCNN(input_shape, signals_list, fusion_type, use_weights)

    elif fusion_type == "learn2combine":
        net = L2C_CNN(input_shape, signals_list, L2C_beta, L2C_delta)


    elif fusion_type == "GBlend":
        net = GBlend_CNN(input_shape, signals_list)


    elif fusion_type == "attention":
        net = AttentionCNN(input_shape, n_branches=n_signals, attention_type="classic")
    elif fusion_type == "selective_fusion":
        net = AttentionCNN(input_shape, n_branches=n_signals, attention_type="selective_fusion")



    elif fusion_type == "decorrelated_classic":
        net = DecorrelatedNet(input_shape, signals_list=signals_list, loss_coef=decorr_loss_weight,
                            plot_conflict=False, cca_type='classic')

    elif fusion_type == "decorrelated_deep":
        net = DecorrelatedNet(input_shape, signals_list=signals_list, loss_coef=decorr_loss_weight,
                            plot_conflict=False, cca_type='deep')




    net = net.to(device)
    print(net)
    return net






#%%
def create_and_train(split, data_type, fusion_type, signals_list,
                     log_power="missing", out_size="missing", interpolation="missing",
                     comp_preprocess_first=False, n_repetitions=1, maxepochs=50,
                     proportion_separate="missing", L2C_beta=0.5, L2C_delta=np.inf,
                     decorr_loss_weight=0.1, use_test=False, plot_results=True):
    """
    create the dataset, the transform and collate functions, and does the
    training of the model


    Parameters
    ----------
    see above for inut type and constraints
    - log_power, out_size, and interpolation are only mandatory when
        data_type == "spectrogram", and can be left ignored otherwise
    - comp_preprocess_first is only mandatory for the late fusions ('probas',
        'scores', and their weighted vatiants), and can be left ignored otherwise
    - comp_preprocess_first is True by default
    - n_repetitions (int) defines how many networks are trained using the same
        protocol, in order to average the results
    - use_test (bool): if True, trains and tests in the conditions of the
        challenge. Default: False
    - plot_results (bool): if False, do not plot the learning curves. The
        confusion matrix is always displayed


    Returns
    -------
    If use_test == False:
        avg_train_F1, std_train_F1, avg_val_F1, std_val_F1  (tuple of floats between 0 and 1)
    If use_test == True:
        avg_train_F1, std_train_F1, avg_val_F1, std_val_F1, avg_test_F1, std_test_F1
    In both cases, if n_repetitions = 1, the standard deviations are zero.

    """
    #return np.random.uniform(0.9, 1), np.random.uniform(0.005,0.015), np.random.uniform(0.7, 0.8), np.random.uniform(0.005,0.015)

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

    if fusion_type in ["probas", "scores", "weighted_probas", "weighted_scores"] and  proportion_separate=="missing":
        raise TypeError("Missing argument: proportion_separate")


    #  -----------------------  preprocessing  -------------------------
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(split, data_type, fusion_type,
                                       signals_list, log_power, out_size, interpolation, comp_preprocess_first, use_test)


    #  ---------------------------  Network training -----------------------------
    list_train_F1, list_val_F1, list_test_F1 = [], [], []



    for _ in range(n_repetitions):
        network = initialize_network(data_type, fusion_type, signals_list, out_size, L2C_beta, L2C_delta, decorr_loss_weight)

        # generate the title the learning curves and confusion matrix
        plot_title = f"""split={split}, data_type={data_type}, fusion_type={fusion_type},
                        signals_list={signals_list},
                        comp_preprocess_first={comp_preprocess_first}, n_repetitions={n_repetitions}, use_test={use_test}"""

        if use_test:
            if fusion_type in ["probas", "scores", "weighted_probas", "weighted_scores"]:
                # then, there is an extra  argument
                _, _, _, train_f1, val_f1, test_f1 = network.test(train_dataloader, val_dataloader, test_dataloader,
                                                                  maxepochs=maxepochs, proportion_separate=proportion_separate)
            else : #"normal" training
                _, _, _, train_f1, val_f1, test_f1 = network.test(train_dataloader, val_dataloader, test_dataloader, maxepochs=maxepochs)
            list_train_F1.append(train_f1)
            list_val_F1.append(val_f1)
            list_test_F1.append(test_f1)

            if plot_results:
                network.plot_confusion_matrix(test_dataloader, plot_title)

        else :
            if fusion_type in ["probas", "scores", "weighted_probas", "weighted_scores"]:
                _, _, train_f1, val_f1 = network.train_process(train_dataloader, val_dataloader,
                                                               maxepochs=maxepochs, proportion_separate=proportion_separate)
            else:
                _, _, train_f1, val_f1 = network.train_process(train_dataloader, val_dataloader, maxepochs=maxepochs)
            list_train_F1.append(train_f1)
            list_val_F1.append(val_f1)

            if data_type == "spectrogram":
                plot_title +=  "\n log_power={}, out_size={}, interpolation={}".format(log_power, out_size, interpolation)

            if plot_results:
                network.plot_learning_curves(plot_title)
                network.plot_confusion_matrix(val_dataloader, plot_title)


    if use_test:
        return np.mean(list_train_F1), np.std(list_train_F1), np.mean(list_val_F1), np.std(list_val_F1),  np.mean(list_test_F1), np.std(list_test_F1)
    else :
        return np.mean(list_train_F1), np.std(list_train_F1), np.mean(list_val_F1), np.std(list_val_F1)





#%% Example
# Possiblities (compatibility is not ensured)
split = ['shuffle', 'balanced', 'unbalanced']

signals = ['Acc_x', 'Acc_y', 'Acc_z', 'Acc_norm',
           'Gyr_x', 'Gyr_y', 'Gyr_z', 'Gyr_norm',
           'Mag_x', 'Mag_y', 'Mag_z', 'Mag_norm',
           'Ori_x', 'Ori_y', 'Ori_z', 'Ori_w', 'Ori_norm']

comp_preprocess_first = [True, False]
    # if True,  apply the preprocessing once, and keep the results in memory
    # if False, apply the preprocesing each time an element is loaded. useful
        # when the whole preprocessed data does not fit into memory

data_type = [ "spectrogram", "temporal", "FFT"]

#for sectrograms
log_power = [True, False]
out_size = (123, 456) # couple of ints,
        # the oupput size of a single spectrogram (before concat)
interpolation = ['linear', 'log', 'none']

fusion_modes_list = ["time", "freq", "depth", "features", "probas", "scores",
               "weighted_probas", "weighted_scores", "GBlend", "learn2combine",
               "selective_fusion", "attention", "decorrelated_classic",
               "decorrelated_deep"]

L2C_beta=0.5  # float between 0 and 1, for the 'learn to combine' paper
L2C_delta=20  # positive float, or np.inf

proportion_separate=0.5 # float, between 0 and 1, only for the late fusion

n_repetitions = 1 # int, the number of times the training is repeated, in
    # order to average the results (preprocessing happens once)

maxepochs = 50
plot_results = [True, False]






process = ["reproduce_Richoz_et_al",
           "reproduce_Ito_et_al",
           "baseline",
           "preprocess_choice",
           "sensor_choice",
           "fusion_mode_choice",
           "proportion_separate_choice",
           "final_model"]






def plot_bar_helper(x, mean_F1, std_F1, color):
    bar = plt.bar(x=x, height=mean_F1, width=1, bottom=0,
                  align="center", color=color, yerr=std_F1,
                  zorder=5)
    # thanks to Greg for the zorder specification
    # stackoverflow.com/questions/23357798/how-to-draw-grid-lines-behind-matplotlib-bar-graph
    return bar



#%%
if __name__ == "__main__":
    process = "fusion_mode_choice"

    # basic network
    if process == "reproduce_Richoz_et_al":
        signals_list = ["Acc_norm", "Gyr_y","Mag_norm", "Ori_w"]

        results = create_and_train(split="balanced", data_type="spectrogram", fusion_type="weighted_scores",
                         signals_list=signals_list, log_power=True, out_size=(48, 48),
                         interpolation="log", comp_preprocess_first=True, maxepochs=50,
                         proportion_separate=0.5,
                         n_repetitions=5, use_test=True, plot_results=True)
        print(results)




    elif process == "reproduce_Ito_et_al":
        # Reproduce the results from ito et al
        signals_list = ['Acc_x', 'Acc_y', 'Acc_z', 'Acc_norm', 'Gyr_x', 'Gyr_y', 'Gyr_z', 'Gyr_norm']
        splits_list = ["shuffle", "unbalanced", "balanced"]
        # we did not know which shuffling they used, so we compared them all
        results = {}

        for split in splits_list:
            for signal in signals_list:
                _,_, avg_val_F1, std_val_F1 = create_and_train(split=split, data_type="spectrogram",
                                           fusion_type="freq", signals_list=[signal],
                                           log_power=True, out_size=(48, 48),
                                           comp_preprocess_first = True,
                                           n_repetitions=5,
                                           interpolation="log", maxepochs=50,
                                           use_test=False, plot_results=False)
                results[(split, signal)] = (avg_val_F1, std_val_F1)



            _,_, avg_val_F1, std_val_F1 = create_and_train(split=split, data_type="spectrogram",
                                 fusion_type="freq", signals_list=['Acc_norm', 'Gyr_y'], log_power=True,
                                 out_size=(48, 48), interpolation="log", comp_preprocess_first=True,
                                 n_repetitions=5, maxepochs=50, use_test=False, plot_results=False)
            results[(split, 'Acc_norm + Gyr_y')] = (avg_val_F1, std_val_F1)

        _,_,_,_, avg_test_F1, std_test_F1 = create_and_train(split="shuffle", data_type="spectrogram",
                             fusion_type="freq", signals_list=['Acc_norm', 'Gyr_y'], log_power=True,
                             out_size=(48, 48), interpolation="log", comp_preprocess_first=True,
                             n_repetitions=5, maxepochs=50, use_test=True, plot_results=False)



        print(9*" " +  "".join(["{:^17s}".format(split)  for split in splits_list]))
        for signal in signals_list + ['Acc_norm + Gyr_y']:
            str_to_print = "{:9s}".format(signal)
            for split in splits_list:
                avg_val_F1, std_val_F1 = results[(split, signal)]
                str_to_print += f"{100*avg_val_F1:5.2f} +/- {100*std_val_F1:.2f} | "
            print(str_to_print)
        print("final, val performance:  {}% +/- {}%".format(avg_val_F1, std_val_F1))
        print("final, test performance: {}% +/- {}%".format(avg_test_F1, std_test_F1))



    if process == "baseline":

        results = create_and_train(split="balanced", data_type="spectrogram", fusion_type="depth",  # fusion does not matter
                         signals_list=["Acc_norm"], log_power=True, out_size=(48, 48),
                         interpolation="log", comp_preprocess_first=True, maxepochs=50,
                         proportion_separate=0.5,
                         n_repetitions=5, use_test=False, plot_results=False)
        train_F1_mean, train_F1_std, val_F1_mean, val_F1_std = results

        results = create_and_train(split="balanced", data_type="spectrogram", fusion_type="depth",  # fusion does not matter
                         signals_list=["Acc_norm"], log_power=True, out_size=(48, 48),
                         interpolation="log", comp_preprocess_first=True, maxepochs=50,
                         proportion_separate=0.5,
                         n_repetitions=5, use_test=True, plot_results=True)
        _,_, _,_, test_F1_mean, test_F1_std = results

        print("---- Baseline ----")
        print(f"Validation perf: {100* val_F1_mean:.2f} ± {100* val_F1_std:.2f} %")
        print(f"Test perf:       {100*test_F1_mean:.2f} ± {100*test_F1_std:.2f} %")





    if process == "final_model":
        signals_list = ["Acc_norm", "Gyr_y", "Ori_w", "Mag_norm"]

        results = create_and_train(split="balanced", data_type="spectrogram", fusion_type="weighted_scores",
                         signals_list=signals_list, log_power=True, out_size=(48, 48),
                         interpolation="log", comp_preprocess_first=True, maxepochs=50,
                         proportion_separate=0.5,
                         n_repetitions=5, use_test=False, plot_results=False)
        train_F1_mean, train_F1_std, val_F1_mean, val_F1_std = results

        results = create_and_train(split="balanced", data_type="spectrogram", fusion_type="weighted_scores",
                         signals_list=signals_list, log_power=True, out_size=(48, 48),
                         interpolation="log", comp_preprocess_first=True, maxepochs=50,
                         proportion_separate=0.5,
                         n_repetitions=5, use_test=True, plot_results=True)
        _,_, _,_, test_F1_mean, test_F1_std = results

        print("---- Final model ----")
        print(f"Validation perf: {100* val_F1_mean:.2f} ± {100* val_F1_std:.2f} %")
        print(f"Test perf:       {100*test_F1_mean:.2f} ± {100*test_F1_std:.2f} %")






#%%
    elif process == "sensor_choice":
        signals_list = ['Acc_x',  'Acc_y',  'Acc_z',  'Acc_norm',
                        'Gra_x',  'Gra_y',  'Gra_z',  'Gra_norm',
                        'LAcc_x', 'LAcc_y', 'LAcc_z', 'LAcc_norm',
                        'Gyr_x',  'Gyr_y',  'Gyr_z',  'Gyr_norm',
                        'Mag_x',  'Mag_y',  'Mag_z',  'Mag_norm',
                        'Ori_x',  'Ori_y',  'Ori_z',  'Ori_w', 'Ori_norm',
                        'Pressure']

        results = {}
        for signal in signals_list:
            _,_, avg_val_F1, std_val_F1 = create_and_train(split="balanced", data_type="spectrogram",
                                       fusion_type="freq", signals_list=[signal],
                                       log_power=True, out_size=(48, 48),
                                       comp_preprocess_first = True,
                                       n_repetitions=5,
                                       interpolation="log", maxepochs=50,
                                       use_test=False, plot_results=False)
            results[signal] = (avg_val_F1, std_val_F1)


        for signal in signals_list:
            avg_val_F1, std_val_F1 = results[signal]
            print(f"{signal:>10s} {100*avg_val_F1:5.2f} +/- {100*std_val_F1:.2f}  ")



        sensors_list = ['Acc', 'Gra', 'LAcc', 'Gyr', 'Ori', 'Mag', 'Pressure']
        colors_per_axis = {'_x':       [0.66, 1.,   0.2],
                           '_y':       [0,    1,    0.6],
                           '_z':       [0.6,  0.8,  0.6],
                           '_w':       [0.2,  0.8,  0.2],
                           '_norm':    [0,    0.6,  0.2],
                           '':   [0.5,  0.8,  0.5]}  # for the 'pressure sensor'
        plt.figure(figsize=(12,5))

        margin = 1 #intersensor margin
        x = 0.5

        legend_dict = {} # dictionnary with same keys as colors_per_axis, and values are matplotlib.artist objecys
        plt.grid('on', axis='y', zorder=0)
        for i_sensor, sensor in enumerate(sensors_list):
            if sensor == 'Ori':
                possible_axis = ['_x', '_y', '_z', '_w', '_norm']
            elif sensor == 'Pressure':
                possible_axis = ['']
            else:
                possible_axis = ['_x', '_y', '_z', '_norm']

            for i_axis, axis in enumerate(possible_axis):
                signal = sensor + axis
                mean, std = results[signal]
                bar = plot_bar_helper(x, 100*mean, 100*std, color=colors_per_axis[axis])
                if axis[1:] not in legend_dict: legend_dict[axis[1:]] = bar
                x += 1.05

            x += margin

        #   legend creation
        del legend_dict['']  # del the pressure
        legend_objects = legend_dict.values()
        legend_names   = legend_dict.keys()
        plt.legend(legend_objects, legend_names, ncol=len(legend_dict), loc='upper center',
                   framealpha=1., borderpad =0.8)

        plt.ylabel('validation F1 (%)', fontsize=13.)
        plt.title("Average and standard deviation over 5 runs")

        n_sensors = len(sensors_list)
        sensor_position_ticks = [ 2.05,  7.25, 12.45, 17.65, 23.35, 29.1, 32.75,]
        full_sensor_names = ['Accelerometer', 'Gravity', 'Linear\nAcceleration', 'Gyrometer',
                             'Orientation\nVector', 'Magnetometer', '   Pressure']
        plt.xticks(sensor_position_ticks, full_sensor_names, fontsize=10.)

        F1_y_ticks = np.linspace(0,100,11).astype(int)
        plt.yticks(F1_y_ticks, F1_y_ticks)

        plt.ylim(30,100)





#%%
    elif process == "preprocess_choice":
        signals_list = ["Acc_norm", "Gyr_y", "Mag_norm", "Ori_w"]

                         #  data_type,  log_power, out_size, interpolation
        preprocess_list = [("spectrogram", True,   None,     'none'),
                           ("spectrogram", False,  None,     'none'),
                           ("temporal",    None,   None,     None),
                           ("FFT",         None,   None,     None),
                           ("spectrogram", True,   (48, 48), 'linear'),
                           ("spectrogram", False,  (48, 48), 'linear'),
                           ("spectrogram", True,   (48, 48), 'log'),
                           ("spectrogram", False,  (48, 48), 'log')]

        results = {}
        for signal in signals_list:
            for preprocess in preprocess_list:
                data_type, log_power, out_size, interpolation = preprocess

                 #comp_preprocess_first = False if ((data_type=="spectrogram") and (interpolation=='none'))  else True

                _,_, avg_val_F1, std_val_F1 = create_and_train(split="balanced", data_type=data_type,
                                           fusion_type="depth", signals_list=[signal],
                                           log_power=log_power, out_size=out_size,
                                           interpolation=interpolation,
                                           comp_preprocess_first = True,
                                           n_repetitions=5, maxepochs=50,
                                           use_test=False, plot_results=False)
                results[(signal, preprocess)] = (avg_val_F1, std_val_F1)

        print("data_type   log_pow out_size   interp " + "".join([f"{signal:^15s} " for signal in signals_list]))
        for preprocess in preprocess_list:
            data_type, log_power, out_size, interpolation = preprocess
            str_to_print = f"{data_type:12s} {str(log_power):6s} {str(out_size):10s} {str(interpolation):7s}"
            for signal in signals_list:
                avg_val_F1, std_val_F1 = results[(signal, preprocess)]
                str_to_print += f"{100*avg_val_F1:5.2f} +/- {100*std_val_F1:.2f} | "
            print(str_to_print)


        n_signals = len(signals_list)


         # plot itself
        colors_per_preprocessing = {
                'temporal':[0.5, 0.5, 1],
                'FFT':[0.2, 0.9, 1],
                'spectrogram_none':[1,0,0],
                'spectrogram_linear':[1,0,0.5],
                'spectrogram_log':[1,0.5,0],
                }

        plt.figure(figsize=(15,6))

        margin = 3 #intersignal margin
        x = 0.5
        empty_artist = matplotlib.patches.Rectangle((0,0), 0,0, edgecolor=[1,1,1,0], facecolor=[1,1,1,0])  # to pad the legends
        legends = []  # will contain couples of (artist, string), these couples will be
                     # put into the list during the frst pass of the loop on signals
        legends.append((empty_artist,''))
        plt.grid('on', axis='y', zorder=0)

        for i_signal, signal in enumerate(signals_list):
            mean, std = results[(signal, ("temporal", None, None, None))] # mean and std are between 0 and 1
            bar = plot_bar_helper(x, 100*mean, 100*std, color=colors_per_preprocessing['temporal'])
            x += 1.5   # we must add at least 1 si that the bars do not overlap,
                       # we add an extra 0.5 for spacing
            if i_signal == 0: legends.append((bar, 'temporal (1D)'))

            mean, std = results[(signal, ("FFT", None, None, None))]
            bar = plot_bar_helper(x, 100*mean, 100*std, color=colors_per_preprocessing['FFT'])
            x += 1.5
            if i_signal == 0: legends.append((bar, 'FFT (1D)'))

            for interpolation in ['none', 'linear', 'log']:
                out_size = None if interpolation == 'none' else (48, 48)
                if i_signal == 0:
                    if interpolation == 'none':
                        interpolation_legend = "full-size spectrogram"
                    else:
                        interpolation_legend = f'small spectrogram, {interpolation:.6s} axis'
                    legends.append((empty_artist, interpolation_legend))

                for log_power in [False, True]:  #['raw_power', 'log_power']:
                    mean, std = results[(signal, ("spectrogram", log_power, out_size, interpolation))]

                    color = colors_per_preprocessing[f'spectrogram_{interpolation}'].copy()
                    if log_power == False: # whiten the color
                        for i in range(3): color[i] = (color[i]+1)/2

                    bar = plot_bar_helper(x, 100*mean, 100*std, color = color)
                    x += 1.05  # extra 0.05 for visibility
                    #if i_signal == 0: legends.append((bar, f'spectrogram, interpolation = {interpolation}, {power}'))

                    legend_name = 'log power' if log_power else 'raw power'
                    if i_signal == 0: legends.append((bar, legend_name))

            x += margin



        # legend creation
        legend_objects = [c[0] for c in legends]
        legend_names   = [c[1] for c in legends]
        plt.legend(legend_objects, legend_names, ncol=4, loc='upper center',
                   framealpha=1., borderpad =0.8)

        plt.ylabel('validation F1 (%)', fontsize=13.)
        plt.title("Average and standard deviation over 5 runs")

        signal_position_ticks = (np.arange(n_signals) +0.5) * (x / n_signals) - margin/2
        plt.xticks(signal_position_ticks, signals_list, fontsize=13.)

        F1_y_ticks = np.linspace(0,100,11).astype(int)
        plt.yticks(F1_y_ticks, F1_y_ticks)

        plt.ylim(0,115)









#%%
    elif process == "fusion_mode_choice":
#        signals_lists_list = [["Acc_norm", "Acc_norm", "Gyr_y"]]
        signals_lists_list = [["Acc_norm", "Gyr_y"],
                              ["Acc_norm", "Mag_norm"],
                              ["Acc_norm", "Gyr_y","Mag_norm"],
                              ["Acc_norm", "Gyr_y","Mag_norm", "Ori_w"]]

        fusion_modes_list = ["time", "freq", "depth", "features", "probas", "scores",
                       "weighted_probas", "weighted_scores", "GBlend", "learn2combine",
                       "selective_fusion", "attention", "decorrelated_classic",
                       "decorrelated_deep"]



        results = {}
        for signals_list in signals_lists_list:
            print("\n\n", signals_list)
            this_result = {}

            for fusion_type in fusion_modes_list:
                if not (len(signals_list) > 2 and fusion_type in ["decorrelated_deep", "decorrelated_classic"]):
                    proportion_separate = 1 if 'probas' in fusion_type else 0.5
                    print("   ", fusion_type)
                    _,_, avg_val_F1, std_val_F1 = create_and_train(split="balanced", data_type="spectrogram",
                                               fusion_type=fusion_type, signals_list=signals_list,
                                               log_power=True, out_size=(48, 48),
                                               comp_preprocess_first = True,
                                               n_repetitions=1,
                                               interpolation="log", maxepochs=50,
                                               proportion_separate=proportion_separate,
                                               decorr_loss_weight=0.1,
                                               use_test=False, plot_results=False)
                    this_result[fusion_type] = (avg_val_F1, std_val_F1)

            results["|".join(signals_list)] = this_result


        for signals_list in signals_lists_list:
            print("\n\n", signals_list)
            for fusion_type in fusion_modes_list:
                avg_val_F1, std_val_F1 = results["|".join(signals_list)][fusion_type]
                print(f"{fusion_type:>18s} {100*avg_val_F1:5.2f} +/- {100*std_val_F1:.2f}  ")

#
#        color_fusion_modes = {  'time concat'            : [0.7,  0. ,  0. ],
#                                'freq concat'            : [0.9,  0. ,  0. ],
#                                'depth concat'           : [0.7,  0.3,  0. ],
#                                'Bottleneck filters' : [0.7,  0.,   0.3],
#                                'features'               : [0.,   0.7,  0. ],
#                                'Selective Fusion'   : [0.6,  1.0,  0.1],
#                                'probas'                 : [0.,   0.5,  0.7],
#                                'scores'                 : [0.5,  0.,   0.7],
#                                'weighted probas'        : [0.2,  0.7,  0.9],
#                                'weighted scores'        : [0.7,  0.2,  0.9],
#                                'Gradient Blend'     : [0.,   0.,   0.9],
#                                'Learn to combine'   : [0.1,  0.1,  0.6],}
        color_fusion_modes = {  'time'            : [0.7,  0. ,  0. ],
                                'freq'            : [0.9,  0. ,  0. ],
                                'depth'           : [0.7,  0.3,  0. ],
                                'bottleneck'       : [0.7,  0.,   0.3],
                                'features'               : [0.,   0.7,  0. ],
                                'selective_fusion'   : [0.6,  1.0,  0.1],
                                'attention'        : [0.5,  0.9,  0.],
                                'probas'                 : [0.,   0.5,  0.7],
                                'scores'                 : [0.5,  0.,   0.7],
                                'weighted_probas'        : [0.2,  0.7,  0.9],
                                'weighted_scores'        : [0.7,  0.2,  0.9],
                                'GBlend'     : [0.,   0.,   0.9],
                                'learn2combine'   : [0.1,  0.1,  0.6],}


        # Plot
        sensor_combination_list = list(results.keys())

        plt.figure(figsize=(21,5))

        margin = 1 #intersensor margin
        x = 0.5

        #legend_dict = {}


        for sensor_combination in sensor_combination_list:

            for fusion_mode in fusion_modes_list:
                mean = 100*np.mean(results[sensor_combination][fusion_mode])
                std =   100*np.std(results[sensor_combination][fusion_mode])

                bar = plot_bar_helper(x, mean, std, color=color_fusion_modes[fusion_mode])
                x += 1.05

            x += margin



        plt.legend(fusion_modes_list, ncol=len(fusion_modes_list)//2,  loc='upper center',
                   framealpha=1., borderpad =0.8)


        plt.ylabel('validation F1 (%)', fontsize=13.)
        plt.title("Average and standard deviation over 5 runs")

        n_sensors = len(sensors_list)

        sensor_combination_position_ticks = (np.arange(len(sensor_combination_list)) +0.5) *1.05* (len(fusion_modes_list)+1)
        full_sensor_combination_names = ['Accelerometer', 'Gravity', 'Linear\nAcceleration', 'Gyrometer']
        plt.xticks(sensor_combination_position_ticks, sensor_combination_list, fontsize=10.)

        F1_y_ticks = np.linspace(0,100,21).astype(int)
        plt.yticks(F1_y_ticks, F1_y_ticks)


        plt.grid('on', axis='y', zorder=0)


        plt.ylim(80,100)
        plt.tight_layout()






#%%
    elif process == "proportion_separate_choice":
        signals_list = ["Acc_norm", "Gyr_y","Mag_norm", "Ori_w"]

        proportion_separate_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]

        late_fusion_modes = ["probas", "scores", "weighted_probas", "weighted_scores", "GBlend"]

        results = {mode:{proportion:() for proportion in proportion_separate_list} for mode in late_fusion_modes}

        for mode in late_fusion_modes:
            for proportion_separate in proportion_separate_list:
                print("\n\n", mode, proportion_separate)

                _,_, avg_val_F1, std_val_F1 = create_and_train(split="balanced", data_type="spectrogram",
                                           fusion_type=mode, signals_list=signals_list,
                                           log_power=True, out_size=(48, 48),
                                           comp_preprocess_first = True,
                                           n_repetitions=5,
                                           interpolation="log", maxepochs=50,
                                           proportion_separate=proportion_separate,
                                           use_test=False, plot_results=False)
                results[mode][proportion_separate] = (avg_val_F1, std_val_F1)



                print('\n\n\n', 10*'_', '\n', mode, proportion_separate, '\n',results, 10*'\n')

        for proportion_separate in proportion_separate_list:
            print(f"{proportion_separate:.1f}: {100*results[proportion_separate][0]} +/- {100*results[proportion_separate][1]}")

        averages = np.array([results[proportion_separate][0] for proportion_separate in proportion_separate_list])
        stds     = np.array([results[proportion_separate][1] for proportion_separate in proportion_separate_list])

        plt.figure()
        plt.plot(proportion_separate_list, averages, c=[0.3, 0.3, 1])
        plt.fill_between(proportion_separate_list, averages-stds, averages+stds, color=[0.3, 0.3, 1, 0.3])
        plt.ylabel("validation F1")
        plt.xlabel("proportions of epochs with separate training")
        plt.title("Average validation F1 ± standard deviation")
        plt.show()
