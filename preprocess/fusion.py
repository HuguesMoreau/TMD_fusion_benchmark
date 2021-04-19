#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains diverse collate functions. Each collate function has two roles :
    - merge together the data from different samples
    - merge together the data from different sensors, when needed

These functions use a list of couples (dictionnary, tensor), where each dictionnary
contains the input data of one sample,


If you are to work with any IPython console (ex: with Jupyter or spyder), is is advised
to launch a '%matplotlib qt' ,to get clean widows
"""


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

import torch
import torch.utils.data

import numpy as np
import matplotlib.pyplot as plt

from param import classes_names, fs, duration_window, duration_overlap, spectro_batch_size, device

from preprocess import Datasets
from preprocess.transforms import TemporalTransform, SpectrogramTransform


#%%
def separate_sensors_collate(L):
    """
    For each sensor, concatenates the signals from each sample into a single matrix.
    Then, return a dict containing one matrix per sensor. The signals can be temporal
    or spectrograms, for the concatenation takes place along a newly-created first
    axis. Also returns a vector of classes.
    One a side note, class reindexing happens here (ie, the input classes are
    between 1 and 8, and the output classes are between 0 and 7)

    /!\ Important note: to perform fusion of vectors, one should not use this
    method, but use a depthwise concatenation. The separation between branches
    of the network will be made using the group argument of nn.Conv layers

    Parameters
    ----------
    L: a list of 2-tuples
        L contains a batch, typically len(L)=64.
        Each tuple is a (eg 60 sec) instance, containing:
            A dict of tensors to be concatenated. Each tensor has a shape of (1, a [,b])
                Keys are sensor names eg "Acc_y", "Acc_norm"
                for raw temporal sensors, the tensor is 2D (1,T) eg T=6000,
                for spectrograms, athe tensor is  (1, F, T)
                NB : here, the shape of the tensor can vary with the sensor.
            A LongTensor with shape(1,) or (6000,)
                (the class, between 1 and 8)


    Returns
    -------
    X_batch_dict : a dict of (B, 1, H [,W]) FloatTensors, containing the input tensors
        from the list L (B = len(L)).
        The dict has the same keys as the dicts in the input list, and still contains one tensor
        per sensor.
        The second axis (with a size of 1) is necessary so that the input data
        has the good shape for Pytorch (channel axis).
    Y_batch : a single tensor containing the B classes (between 0 and 7)

    """
    batch_size = len(L)
    x_example = L[0][0] # a dict relative to first instance
    y_example = L[0][1]
    signal_name_list = x_example.keys() # all dicts in L are assumed to have the same keys

    # Allocation
    X_batch_dict = {}
    for signal_name in signal_name_list:
         shape = (batch_size,) + x_example[signal_name].shape
         X_batch_dict[signal_name] = torch.zeros(shape, dtype=torch.float32, device=device)

    Y_batch = torch.zeros(batch_size, dtype=y_example.dtype, device=device)

    # Filling
    for index_batch, (instance, y) in enumerate(L):

        for signal_name in signal_name_list:
            X_batch_dict[signal_name][index_batch,0,...] = instance[signal_name] # instance[signal_name] is 1 or 2D array

        Y_batch[index_batch] = y

    # -1 because in txt file, label is 1..8 and Pytorch need a class from 0..(n-1)
    Y_batch = Y_batch-1

    return X_batch_dict, Y_batch







#%%
class ConcatCollate():
    def __init__(self, axis_collate, list_signals=None, flag_debug=False):
        """
        Create a collate function using the provided axis to concatenate the input tensors

        Parameter
        ---------
        axis_collate (string): either 'depth', 'time', or 'freq'
        list_signals (list, optionnal): if we want to use the same signal
            several times, add the names in a list. Examples
            ["Acc_norm", "Acc_norm"] to use the norm of accelerometer twice
            ["Acc_norm", "Gyr_y", "Gyr_y", "Gyr_y", "Mag_norm"]
            Defaults to None (in which case each signla is used once)
        flag_debug (bool): whether to printdebugging messages



        Attributes
        -------
        self.axis_concat (string): either 'depth', 'time', or 'freq'
        self.flag_debug
        self.list_signals

        """

        if axis_collate in ['depth', 'time', 'freq']:
            self.axis_concat = axis_collate
        else :
            error_message = "unknown axis for concatenation: {}. \n Choose either 'depth', 'time', or 'freq'".format(axis_collate)
            raise np.AxisError(error_message)

        self.flag_debug = flag_debug
        self.list_signals = list_signals


    def __call__(self, L):
        """
        1. Concatenate spectrograms along a specified axis, and
        2. MERGE into a batch
        The axis is given by the self.axis variable
        One a side note, class reindexing happens here (ie, the input classes are
        between 1 and 8, and the output classes are between 0 and 7)


        Parameters
        ----------
        L: a list of 2-uples,
            L contains a batch, typically len(L)=64.
            Each tuple is a (eg 60 sec) instance, containing:
                A dict of tensors to be concatenated. Each tensor has a shape of (1, a [,b])
                    Keys are sensor names eg "Acc_y", "Acc_norm"
                    for raw temporal sensors, the tensor is 2D (1,T) eg T=6000,
                    for spectrograms, athe tensor is  (1, F, T)
                    /!\ The arrays must have the same shape
                A LongTensor with shape(1,) (the class, between 1 and 8)


        Returns
        -------
        X : a single 3-dim or 4-dim FloatTensor
        Y : a single tensor containing all the classes (between 0 and 7)

        Shape of the output tensor X :

                        data type ->  |       temporal          |      spectrogram       |
        concat type :                 |        or FFT           |                        |
        ----------------------------------------------------------------------------------
                    depth             |       (B, S, T)         |    (B,  S,  F,  T )    |
        ----------------------------------------------------------------------------------
                    time              |      (B, 1, S*T)        |    (B,  1,  F, S*T)    |
        ----------------------------------------------------------------------------------
                    freq              |      (B, 1, S*T)        |    (B,  1, S*F, T )    |

        With B: batch size (==len(L)) , S: number of signals, T: number of timesteps, F: number of frequencies

        Remark: as both temporal and FFT signals are 1-dimensional, nothing
        prevents the user to concatenate temporal signals along their 'frequency'
        axis, and this will lead to the same results as if the 'time' axis was
        asked. We assume the user knows what they are doing.
        Same goes for FFT signals with 'time' concatenation.
        """

        if self.list_signals == None:
            # take the signals of the first sample
            signal_name_list = list(L[0][0].keys())
        else:
            signal_name_list = self.list_signals

        batch_size = len(L)

        x_example = L[0][0] # a dict relative to first instance
        y_example = L[0][1]
#        signal_name_list = list(x_example.keys())  # list of sensor name
        n_signals = len(signal_name_list)
        signal_name_example = signal_name_list[0] # string

        instance = x_example[signal_name_example] # array (1,n) or (1,nb_f, nb_t)
        data_type = '1D' if len(instance.shape)==2 else '2D'
            # 1D = Temporal or fft
            # 2D = spectrograms

        flag_debug = self.flag_debug

        # Get an index for the dimension to concatenate along
        if self.axis_concat == 'depth':
            axis_concat = 0
        elif self.axis_concat == 'time':
            if data_type == '1D':
                axis_concat = 1
            elif data_type == '2D':
                axis_concat = 2
        elif self.axis_concat == 'freq':
            if data_type == '1D':
                axis_concat = 1
            elif data_type == '2D':
                axis_concat = 1

        signal_shape = instance.shape
        batch_shape = (batch_size,) + signal_shape
        batch_shape = list(batch_shape)  # tuples cannot be modified in-place
        batch_shape[axis_concat+1] *= n_signals   # for the final tensor, the first axis is devoted to the batch, hence the +1
        batch_shape = tuple(batch_shape)

        if flag_debug:
            print('X_batch shape')
            print(batch_shape)

        X_batch = torch.zeros(batch_shape, dtype=torch.float32, device=device)
        Y_batch = torch.zeros(batch_size,  dtype=y_example.dtype, device=device)

        for index_batch, (instance, y) in enumerate(L):

            signal_list = [instance[signal_name] for signal_name in signal_name_list] # all signals for the given instance

            torch.cat( signal_list, dim=axis_concat, out=X_batch[index_batch,...] )

            Y_batch[index_batch] = y

        # sending the data to cuda
        Y_batch = Y_batch-1
        return X_batch, Y_batch



#%% collate test
if __name__ == "__main__":

    """
        test the online (comp_preprocess_first=False) preprocessing
        load the Train Set
        apply the preprocess on the first 5 samples

    """

    print('\n\n *** test the online (comp_preprocess_first=False) preprocessing *** \n')

    n_classes = len(classes_names)
    # We will need this for the tests
    DS = Datasets.SignalsDataSet(mode='train', split='balanced', comp_preprocess_first=False)


    flag_debug = True
    example_signals = ["Acc_norm", "Gyr_y", "Mag_norm"]
    n_signals = len(example_signals)


    # ---------------------- temporal ----------------------------
    temporal_transform = TemporalTransform(example_signals)
    DS.transform = temporal_transform
    dataloader = torch.utils.data.DataLoader(DS, batch_size=5) # instances will be loaded 5 by 5

    plt.figure()

    #                        axis = time
    collate_concat = ConcatCollate(axis_collate='time', flag_debug=flag_debug)
    dataloader.collate_fn = collate_concat

    X_batch, _ = next(iter(dataloader))  # thanks to https://github.com/pytorch/pytorch/issues/1917
    # we load only once

    signal = X_batch[0,0,:]
    plt.subplot(3,1,1) # thanks to https://stackoverflow.com/questions/2265319/how-to-make-an-axes-occupy-multiple-subplots-with-pyplot-python

    plt.plot(signal.to(torch.device('cpu')).numpy())
    plt.title('time concat')


    #                        axis = depth
    collate_concat = ConcatCollate(axis_collate='depth')
    dataloader.collate_fn = collate_concat
    X_batch, _ = next(iter(dataloader))

    for i_signal, signal_name in enumerate(example_signals):
        plt.subplot(3,n_signals,n_signals+i_signal+1)
        plt.title(example_signals[i_signal]+ '(form depth)')
        signal = X_batch[0,i_signal,:]
        plt.plot(signal.to(torch.device('cpu')).numpy())
    plt.show()




    #                        axis = freq
    collate_concat = ConcatCollate(axis_collate='freq')
    dataloader.collate_fn = collate_concat
    X_batch, _ = next(iter(dataloader))

    signal = X_batch[0,0,:]
    plt.subplot(3,1,3)
    plt.plot(signal.to(torch.device('cpu')).numpy())
    plt.title("'frequency' concat")



    # ---------------------- spectrogram ----------------------------
    spectrogram_transform = SpectrogramTransform(example_signals, fs, duration_window, duration_overlap, spectro_batch_size,
                                                 interpolation='none', log_power=True)
    DS.transform = spectrogram_transform


    #                        axis = freq
    plt.figure()
    collate_concat = ConcatCollate(axis_collate='freq')
    dataloader.collate_fn = collate_concat

    X_batch, _ = next(iter(dataloader))

    signal = X_batch[0,0,:,:]

    plt.subplot(1,2,1)
    plt.imshow(signal.to(torch.device('cpu')).numpy())
    plt.title('freq concat')
    plt.colorbar()

    #                        axis = depth
    collate_concat = ConcatCollate(axis_collate='depth')
    dataloader.collate_fn = collate_concat
    X_batch_concat, _ = next(iter(dataloader))

    for i_signal, signal_name in enumerate(example_signals):
        plt.subplot(n_signals, 2, 2*(i_signal+1))
        plt.title(example_signals[i_signal])
        signal = X_batch_concat[0,i_signal,:]
        plt.imshow(signal.to(torch.device('cpu')).numpy())
        plt.colorbar()

    plt.show()


    #                        axis = time
    plt.figure()
    collate_concat = ConcatCollate(axis_collate='time')
    dataloader.collate_fn = collate_concat

    X_batch, _ = next(iter(dataloader))

    signal = X_batch[0,0,:,:]
    plt.subplot(2,1,1)
    plt.imshow(signal.to(torch.device('cpu')).numpy())
    plt.colorbar()
    plt.title('time concat')

    #                        axis = depth  (for visualization purposes)
    # no need to reload X_batch_concat
    for i_signal, signal_name in enumerate(example_signals):
        plt.subplot(2, n_signals, n_signals+i_signal+1)
        plt.title(example_signals[i_signal])
        signal = X_batch_concat[0,i_signal,:]
        plt.imshow(signal.to(torch.device('cpu')).numpy())
        plt.colorbar()

    plt.show()



    # signal duplication test
    plt.figure()
    collate_concat = ConcatCollate(axis_collate='time', list_signals=["Acc_norm", "Acc_norm", "Gyr_y"])
    dataloader.collate_fn = collate_concat

    X_batch, _ = next(iter(dataloader))

    signal = X_batch[0,0,:,:]
    plt.subplot(2,1,1)
    plt.imshow(signal.to(torch.device('cpu')).numpy())
    plt.colorbar()
    plt.title('time concat ["Acc_norm", "Acc_norm", "Gyr_y"]')

    #                        axis = depth  (for visualization purposes)
    # no need to reload X_batch_concat
    for i_signal, signal_name in enumerate(example_signals):
        plt.subplot(2, n_signals, n_signals+i_signal+1)
        plt.title(example_signals[i_signal])
        signal = X_batch_concat[0,i_signal,:]
        plt.imshow(signal.to(torch.device('cpu')).numpy())
        plt.colorbar()

    plt.show()

    del DS




#%%
# =============================================================================
#       Complete test : using DataLoader and comp_preprocess_first = True
# =============================================================================

if __name__ == "__main__":

    """
    """

    print('\n\n *** test the OFFLINE (comp_preprocess_first=True) preprocessing on Validation SET *** \n')


    temporal_transform = TemporalTransform(["Acc_norm", "Gyr_y", "Mag_norm", "Ori_w"])
    spectrogram_transform = SpectrogramTransform(["Acc_norm", "Gyr_y", "Mag_norm", "Ori_w"], fs,
                                                 duration_window, duration_overlap, spectro_batch_size,
                                                 interpolation='log', log_power=True, out_size=(48,48))
    plt.figure()
    index_plot = 1

    for transform_to_apply in [temporal_transform, spectrogram_transform]:
        DS_transformed = Datasets.SignalsDataSet(mode='val', split='balanced',
                                    comp_preprocess_first=True, transform = transform_to_apply)

#        import platform
#        num_workers = 0 if platform.system() == 'Windows' else 5
#        # I couldn't make multithreading work on Windows, so I deactivate it
        dataloader = torch.utils.data.DataLoader(DS_transformed, batch_size=5, collate_fn=separate_sensors_collate)

        batch = next(iter(dataloader))
        x_batch = batch[0]
        y_first_sample = batch[1][0].item()
        signal_name_list = x_batch.keys()

        n_signals = len(signal_name_list)

        for signal_name in signal_name_list:
            plt.subplot(2, n_signals, index_plot)
            index_plot += 1
            plt.title("Dataloader test: first element (index=0)\n signal: {}, class: {}".format(signal_name, classes_names[y_first_sample]))
            if transform_to_apply == temporal_transform:
                batch_signal = x_batch[signal_name]
                plt.plot(batch_signal[0,0,:].cpu().numpy())
            if transform_to_apply == spectrogram_transform:
                batch_spectrogram = x_batch[signal_name]
                plt.imshow(batch_spectrogram[0,0,:,:].cpu().numpy())

        plt.show()

        del DS_transformed