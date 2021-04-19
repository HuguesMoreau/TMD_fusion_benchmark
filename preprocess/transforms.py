#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains diverse preprocessing functions (mostly norms ans spectrograms),
and basic tests and visualizations.
If you are to work with any IPython console (ex: with Jupyter or spyder), is is advised
to launch a '%matplotlib qt' ,to get clean widow
"""


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

import numpy as np
import torch
import scipy.signal, scipy.interpolate, scipy.ndimage


from param import classes_names, fs, duration_window, duration_overlap, duration_segment, spectro_batch_size
from preprocess import Datasets

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n_classes = len(classes_names)
    # We will need this for the tests
    DS = Datasets.SignalsDataSet(mode='train', split='balanced', comp_preprocess_first=False)


#%% transform functions

"""In all following functions, the input parameter (data) is, by default,
 a dict of numpy arrays, containing signal names (eg. "Gyr_z") as keys, and 1-dimensional
 arrays as values

Most of this part contains basic visualizations to make sure the preprocessing is correct"""




class TemporalTransform():
    """  create the base transform to use to each element of the data
    Also generates data for thr 'Random' and 'Zero' cases

    Parameters
    ----------
    signal_name_list: a list of string signals (ex: 'Gyr_y', 'Ori_x')
        If a string ends by "_norm" (ex: "Mag_norm"), the output will
        be the norm of the three (or four) axis of the signal.
        The signals can also be 'Zero', in which case the segments are only
        zeros, or 'Random', in which case each data oint is sampled with normal
        distribution (zero mean, unit variance)

    Returns
    -------
    a function with input:  a dict of (_, 6000) arrays (key example: 'Gyr_y')
                and output: a dictionnary of arrays.
    """
    def __init__(self, signal_name_list):
        super(TemporalTransform, self).__init__()
        self.signal_name_list = signal_name_list


    def __call__(self, data):
        """
        Parameters
        ----------
        a dict of (B, 6000) arrays (key example: 'Gyr_y')

        Returns
        -------
        a dictionnary of arrays. This time, the keys are from signal_name_list,
            and the values are either raw signals (if the key ends with '_x',
            '_y', '_z', or '_w'); a norm of several signals (if the key ends
            with '_norm'); or a specific signal (if the key is 'Random' or
            'Zero'). The shape of each array is (B, 6000), where B (batch size)
            depends on the input shape.
        """

        outputs = {}
        for signal_name in self.signal_name_list:

            if signal_name[-2:] in ['_x', '_y', '_z', '_w'] or signal_name == "Pressure":
                processed_signal = data[signal_name]

            elif signal_name == 'Random':
                data_shape = data["Acc_x"].shape
                processed_signal = np.random.randn(data_shape[0], data_shape[1]).astype(np.float32)

            elif signal_name == 'Zero':
                data_shape = data["Acc_x"].shape
                processed_signal = np.zeros(data_shape).astype(np.float32)


            elif signal_name[-5:] == '_norm':
                suffix_location = signal_name.index("_") # 4 if signal_name == "LAcc", 3 otherwise
                sensor = signal_name[:suffix_location]    # ex: 'Acc', 'LAcc'

                if sensor == "Ori":
                    # in that case, data[sensor+"_x"]**2 + data[sensor+"_y"]**2 + data[sensor+"_z"]**2 should be 1.0
                    processed_signal = np.sqrt(data[sensor+"_x"]**2 + data[sensor+"_y"]**2 + data[sensor+"_z"]**2 \
                                             + data[sensor+"_w"]**2)
                else :
                    processed_signal = np.sqrt(data[sensor+"_x"]**2 + data[sensor+"_y"]**2 + data[sensor+"_z"]**2)
            else :
                raise ValueError("unknown signal name: '{}'. Signal names should end with either '_x', '_y', '_z', '_w', or '_norm'".format(signal_name))

            outputs[signal_name] = processed_signal

        return outputs



    def __str__(self):
        """purely for visual purposes, so that we can print() the function"""
        str_to_return = "Temporal_transform"
        str_to_return += "\n\t Signals: {}".format(self.signal_name_list)
        return str_to_return






if __name__ == "__main__":

    # plot one figure per sensor
    # on each figure, one subplot per class,
    # to find one instance per each class, we start looking at index = index0
    index0 = 0

    for tested_signal_name in ["Acc_norm", "Ori_norm", "Mag_norm", "LAcc_x"]:
        # plot 1 segment from each class.
        plt.figure()

        if tested_signal_name != 'Pressure':
            suffix_location = tested_signal_name.index("_")
            tested_sensor = tested_signal_name[:suffix_location]    # ex: 'Acc', 'LAcc'
        else:
            tested_sensor = 'Pressure'

        sensor_axis = [tested_sensor + axis for axis in ["_x", "_y", "_z"]] if tested_sensor != 'Pressure' else ['Pressure']
        if tested_sensor == "Ori" :  sensor_axis.append(tested_sensor+"_w")

        temporal_transform = TemporalTransform([tested_signal_name])

        remaining_classes = classes_names.copy()
        index = index0

        while len(remaining_classes)>0:
            data_tensor, class_tensor = DS[index] # data is a dict of 2D tensors (1,nb)
            data_cpu = {signal:data_tensor[signal].to(torch.device('cpu')).detach().numpy() for signal in data_tensor.keys()}
            class_index = int(class_tensor)
            class_name = classes_names[class_index-1]

            if class_name in remaining_classes:

                remaining_classes.remove(class_name)
                plt.subplot(2, 4, n_classes - len(remaining_classes))


                for k,signal in enumerate(sensor_axis):

                    if k==0:  # compute the temporal axis once
                        nb = data_cpu[signal].shape[1]
                        x_t = np.linspace(0, nb/fs, nb)

                    plt.plot(x_t, data_cpu[signal][0,:])

                selected_signal = temporal_transform(data_cpu)[tested_signal_name]

                error_message_dtype = "One of the signals does not have the correct type: {}, {} \n dtype should be float32, is actually {}".format(tested_signal_name, str(temporal_transform), selected_signal.dtype)
                assert (selected_signal.dtype == 'float32'), error_message_dtype

                plt.plot(x_t, selected_signal[0,:], '--')
                plt.xlabel("t (s)")
                legend = sensor_axis + [tested_signal_name+' (selected)']
                plt.legend(legend)
                plt.title("{} ({}, index={})".format(tested_sensor, classes_names[class_index-1], index))


            index +=1

        plt.show()




#%% FFT

class FFTTransform():
    """  create a transform to use to return the power of the spectrum
    (computed through a Fourier transform) of each element of the data

    Parameters
    ----------
    signal_name_list: a list of string signals (ex: 'Gyr_y', 'Ori_x')
        If a string ends by "_norm" (ex: "Mag_norm"), the output will
        be the norm of the three (or four) axis of the signal.

    Returns
    -------
    a function with input:  a dict of (B, 6000) arrays (key example: 'Gyr_y')
                and output: a dictionnary of (B, 6000) arrays.
    """
    def __init__(self, signal_name_list):
        super(FFTTransform, self).__init__()
        self.signal_name_list = signal_name_list
        self.temporal_transform = TemporalTransform(signal_name_list)


    def __call__(self, data):
        """
        Parameters
        ----------
        a dict of (B, 6000) arrays (key example: 'Mag_x')

        Returns
        -------
        a dictionnary of arrays. The keys are from signal_name_list, and the
            values are the power spectra of each signal. The shape of each array is
            (B, 6000), where B (batch size) depends on the input shape
        """
        temporal_signals = self.temporal_transform(data)
        del data  # free some memory

        outputs = {}
        for signal_name in self.signal_name_list:
            complex_fft = np.fft.fft(temporal_signals[signal_name], axis=1)
            power_fft = np.abs(complex_fft)
            power_fft[:,0] = 0. # remove the DC component (to avoid this component
              #  outscales te others)

            centered_power_fft = np.fft.fftshift(power_fft, axes=1) # so 0 Hz is in the middle

            outputs[signal_name] = centered_power_fft.astype('float32')
            del temporal_signals[signal_name] # release the memory

            # a faire, calculer les f et les sauver
            # self.f=f

        return outputs



    def __str__(self):
        """purely for visual purposes, so that we can print() the function"""
        str_to_return = "FFT_transform"
        str_to_return += "\n\t Signals: {}".format(self.signal_name_list)
        return str_to_return




#%%

if __name__ == "__main__":

    # classes to plot,
    sel_classes = ["Still","Walk","Run","Train"]
    nsel = len(sel_classes)

    for tested_signal_name in ["Acc_norm", "Gyr_y", "Mag_norm", "Pressure"]:
        # plot 1 segment from each class.
        plt.figure()
        tested_sensor = tested_signal_name[:3]
        if "_" in tested_sensor:
            sensor_axis = [tested_sensor + axis for axis in ["_x", "_y", "_z"]]
        else: # Pressure
            sensor_axis = [tested_sensor]
        if tested_sensor == "Ori" :  sensor_axis.append(tested_sensor+"_w")

        fft_transform =      FFTTransform([tested_signal_name])
        temporal_transform = TemporalTransform([tested_signal_name])

        remaining_classes = sel_classes.copy()

        index = 0
        isub = 1

        while len(remaining_classes)>0:
            data_tensor, class_tensor = DS[index]
            data_cpu = {signal:data_tensor[signal].to(torch.device('cpu')).detach().numpy() for signal in data_tensor.keys()}
            class_index = int(class_tensor)

            class_name = classes_names[class_index-1]

            if class_name in remaining_classes:
                remaining_classes.remove(class_name)

                plt.subplot(2, nsel, isub)
                selected_signal = temporal_transform(data_cpu)[tested_signal_name]

                 # plot the temporal signal
                nb = selected_signal.shape[1]
                x_t = np.linspace(0, nb/fs, nb)

                x_f = np.linspace(-fs/2,fs/2, nb)

                plt.plot(x_t, selected_signal[0,:])
                plt.xlabel("t (s)")
                plt.title("{} ({}, index={})".format(tested_signal_name, classes_names[class_index-1], index))

                 # plot the fft
                plt.subplot(2, nsel, isub + 4)
                isub += 1
                selected_power = fft_transform(data_cpu)[tested_signal_name]

                error_message_dtype = "One of the signals does not have the correct type: {}, {} \n dtype should be float32, is actually {}".format(tested_signal_name, str(fft_transform), selected_power.dtype)
                assert (selected_power.dtype == 'float32'), error_message_dtype

                plt.plot(x_f, selected_power[0,:])
                plt.xlabel("f (Hz)")
                plt.title("FFT of {} ({}, index={})".format(tested_signal_name, classes_names[class_index-1], index))

            index +=1
        plt.show()








#%%

#  ----------------  Spectral transforms  ---------------------


# Interpolation functions
def interpol_log(f, t, spectrogram, out_size):
    """interpolates the spectrogram in input using a linear axis for the timestamps and a LOG axis for the frequencies

    Parameters
    ----------
    f : numpy array, shape: (F_in,), frequencies of the spectrogram
    t : numpy array, shape: (T_in,), timestamps of the spectrogram
    spectrogram : (B, F_in, T_in), B is batch size; 3D numpy array

    out_size : couple of ints (F_out, T_out)

    Returns
    -------
    f_interpolated : numpy array, shape: (F_out,), frequencies of the spectrogram AFTER interpolation
    t_interpolated : numpy array, shape: (T_out,), timestamps of the spectrogram AFTER interpolation
    a spectrogram, where the f axis (second dimension) has been re-interpolated
    using a log axis

    """
    B = spectrogram.shape[0]
    out_f, out_t = out_size

    log_f = np.log(f+f[1]) #  log between 0.2 Hz and 50.2 Hz

    log_f_normalized    = (log_f-log_f[0])/(log_f[-1]-log_f[0]) # between 0.0 and 1.0
    t_normalized        = (t-t[0])/(t[-1]-t[0])

    rescaled_f = out_f*log_f_normalized # 0 and 48
    # rescaled_f = (out_f-1)*log_f_normalized ??
    rescaled_t = out_t*t_normalized

    spectrogram_interpolated = np.zeros( (B, out_f, out_t), dtype='float32')
    index_f, index_t = np.arange(out_f), np.arange(out_t) # between 0 and 47

    for i in range(B):
        spectrogram_fn = scipy.interpolate.interp2d(rescaled_t, rescaled_f, spectrogram[i,:,:], copy=False)
        # interp2d returns a 2D function
        spectrogram_interpolated[i,:,:] = spectrogram_fn(index_t, index_f)  # care to the order

    f_fn = scipy.interpolate.interp1d(rescaled_f, f, copy=False)
    f_interpolated = f_fn(index_f)

    t_fn = scipy.interpolate.interp1d(rescaled_t, t, copy=False)
    t_interpolated = t_fn(index_t)


    return f_interpolated, t_interpolated, spectrogram_interpolated



def interpol_lin(f, t, spectrogram, out_size):
    """interpolates the spectrogram in input using a linear axis for the timestamps AND the frequencies

    Parameters
    ----------
    f : numpy array, shape: (F_in,), frequencies of the spectrogram
    t : numpy array, shape: (T_in,), timestamps of the spectrogram
    spectrogram : (B, F_in, T_in) numpy array
    out_size : couple of ints (F_out, T_out)
     (does not need f or t)


    Returns
    -------
    f_interpolated : numpy array, shape: (F_out,), frequencies of the spectrogram AFTER interpolation
    t_interpolated : numpy array, shape: (T_out,), timestamps of the spectrogram AFTER interpolation
    a spectrogram: 3D numpy array, where the f axis (second dimension) has been re-interpolated
          using a linear axis
    """
    B, F_in, T_in = spectrogram.shape
    out_f, out_t = out_size
    output_shape = (B, out_f, out_t ) # result is (B, out_f, out_t )

    rescale_factor_d = 1.   # for depth
    rescale_factor_f = F_in/out_f # typically 550/48
    rescale_factor_t = T_in/out_t

    matrix_transform = np.diag( np.array([rescale_factor_d, rescale_factor_f, rescale_factor_t]) ) # (3,3) matrix

    # spectrogram = matrix_transform * spectrogram_interpolated
    spectrogram_interpolated = scipy.ndimage.affine_transform(spectrogram, matrix_transform, offset=0, order=1, output_shape=output_shape)
        # we only use linear interpolation because we almost always downsample, and because 2nd order methods and above
        # have a nasty tendency to create small negative local minimas between two strictly positive values
        # we do not want this when we apply a log to the values of the spectrogram

    f_interpolated = scipy.ndimage.affine_transform(f, np.array( [rescale_factor_f] ) , offset=0, order=1, output_shape = (out_f,) )
    t_interpolated = scipy.ndimage.affine_transform(t, np.array( [rescale_factor_t] ) , offset=0, order=1, output_shape = (out_t,) )


    return f_interpolated, t_interpolated, spectrogram_interpolated



def no_interpolation(f, t, spectrogram, out_size):
    """ This function is just a placeholder that mimics the arguments
    of the two previous interpolation functions    """
    return f, t, spectrogram










#%%
#  ---------------- The spectrogram class --------------
class SpectrogramTransform():
    """ create the transform to work with spectrograms. This class behaves
    essentially the same as TempralTransform, except the created transform
    returns a dict of 3d array instead of 2d


    Parameters
    ----------
    signal_name_list: a list of string signals (ex: 'Gyr_y', 'Ori_x')
        If a string ends by "_norm" (ex: "Mag_norm"), the output will
        be the norm of the three (or four) axis of the signal.
    fs: sampling frequency
    duration_window, duration_overlap: duration in sec of spectrogram window and overlap
    spectro_batch_size:
        turning 13,000 temporal signals into (550, 500) array
        spectrograms at once is too much: a single (13000, 550, 500) array,
        with simple precision requires 7.15 Go !
        This is why we work with batches of 1000 instead. For each batch,
        we compute the complete sectrogram (1000 x 550 x 500), then
        interpolate it to smaller sizes, before working wit the following batch.

    interpolation :  string ("log", "linear", "none")
    log_power : bool. If True, the values of the  power spectrum are replaced
        by their log
    out_size : tuple of integer (nb_interp_f, nb_interp_t) = size of spectrogram AFTER interpolation        Is ignored if no interpolation occurs. Default: None
        the spectrogram is computed for 2 1D-arrays: f and t

    flag_debug: flag for print debugging info



    Returns
    -------
    a function with input: data : a dict of (_, 6000) arrays  (key example: 'Gyr_y')
                and output: a dictionnary of 2d arrays.

    """
    def __init__(self, signal_name_list, fs, duration_window, duration_overlap, spectro_batch_size, interpolation,
                 log_power, out_size=None, flag_debug=False):
        super(SpectrogramTransform, self).__init__()

        self.temporal_transform = TemporalTransform(signal_name_list)
        self.fs = fs
        self.duration_window = duration_window
        self.duration_overlap = duration_overlap
        self.spectro_batch_size = spectro_batch_size

        self.signal_name_list = signal_name_list
        self.log_power = log_power
        self.interpolation_name = interpolation

        if interpolation == "linear":
            self.interpolation_fn = interpol_lin
            self.out_size = out_size

        elif interpolation == "log":
            self.interpolation_fn = interpol_log
            self.out_size = out_size

        elif interpolation == "none":
            self.interpolation_fn = no_interpolation
            self.out_size = None

        else :
            raise ValueError("Unknown interpolation: '{}'. Use one of 'log', 'linear', 'none'".format(interpolation))

        # if interpolation == "none" and out_size != None :
        #     warnings.warn("No interpolation is to take place, but an target output size was provided. the output_size argument will be ignored",  Warning)

        self.flag_debug = flag_debug

    def __call__(self, data):
        """
        Parameters
        ----------
        data : a dict of (B, 6000) arrays  (key example: 'Gyr_y')

        Returns
        -------
        a dictionnary of 2d arrays. The keys are from signal_name_list,
            and the values are either spectrograms of raw signals (if the key
            ends with '_x', '_y', '_z', or '_w'); or a spectogram of a norm of
            signals (if the key ends with '_norm'). The shape of the spectrogram
            is (B, F, T), where B (batch size) depends on the input shape, and
            F and T are given by self.out_size
        """


        temporal_signals = self.temporal_transform(data)
        del data  # free some memory


        fs = self.fs

        nperseg     = int(self.duration_window * fs)
        noverlap    = int(self.duration_overlap * fs)

        spectro_batch_size = self.spectro_batch_size
        # turning 13,000 temporal signals into (550, 500) array
            # spectrograms at once is too much: a single (13000, 550, 500) array,
            # with simple precision requires 7.15 Go !
            # This is why we work with batches of 1000 instead. For each batch,
            # we compute the complete sectrogram (1000 x 550 x 500), then
            # interpolate it to smaller sizes, before working wit the following batch.

        out_size = self.out_size

        flag_debug = self.flag_debug

        outputs = {}

        for signal_name in self.signal_name_list:
            current_spectro_batch_size = temporal_signals[signal_name].shape[0]

            if current_spectro_batch_size < spectro_batch_size :
                f, t, spectrogram = scipy.signal.spectrogram(temporal_signals[signal_name], fs=fs, nperseg=nperseg, noverlap=noverlap)

                f_interpolated, t_interpolated, interpolated_spectrogram = self.interpolation_fn(f, t, spectrogram, out_size)
                        # f, t, and possibly out_size will be ignored when the function does not need them

            else :
                n_batches = (current_spectro_batch_size-1)//spectro_batch_size +1



                if out_size is not None:  # we actually compute the interpolation
                    nb_interp_f, nb_interp_t = out_size

                else:    # we only recompute the shapes of the raw spectrogram
                    nb_interp_f = int(duration_window*fs/2) +1
                    nb_interp_t = int((duration_segment-duration_window)/(duration_window-duration_overlap)) +1

                interpolated_spectrogram = np.zeros((current_spectro_batch_size, nb_interp_f, nb_interp_t), dtype='float32')
                for i in range(n_batches):
                    i_min =   i   * spectro_batch_size
                    i_max = (i+1) * spectro_batch_size  # does not matter if it goes beyond current_spectro_batch_size
                    this_temporal_signal = temporal_signals[signal_name][i_min:i_max,:]

                    f, t, spectrogram = scipy.signal.spectrogram(this_temporal_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

                    if out_size is not None:
                        f_interpolated, t_interpolated, interpolated_spectrogram[i_min:i_max,:,:] = self.interpolation_fn(f, t, spectrogram, out_size) # erase the spectrogram by its interpolation
                    else:
                        f_interpolated, t_interpolated, interpolated_spectrogram[i_min:i_max,:,:] = f, t, spectrogram


            if flag_debug:
                print('f_interpolated:')
                print(f_interpolated[:5])
                print(f_interpolated[-5:])


            del temporal_signals[signal_name]

            if self.log_power :
                np.log(interpolated_spectrogram + 1e-10, dtype='float32', out=interpolated_spectrogram)

            outputs[signal_name] = interpolated_spectrogram

            self.f_interpolated = f_interpolated
            self.t_interpolated = t_interpolated

            # for future debug
            # self.f = f
            # self.t = t
            # self.spectrogram = spectrogram




        return outputs



    def __str__(self):
        """purely for visual purposes, so that we can print() the function"""
        str_to_return = "Spectrogram transform"
        str_to_return += "\n\t Signals: {}".format(self.signal_name_list)
        if self.interpolation_fn != no_interpolation:
            if self.out_size != None :
                str_to_return += "\n\t Output size: {}".format(self.out_size)
            else :
                str_to_return += "\n\t Output size: unchanged"

        str_to_return += "\n\t Interpolation: {}".format(self.interpolation_name)
        str_to_return += "\n\t Log power: {}".format(self.log_power)

        return str_to_return

# end of class SpectrogramTransform():



#%%
if __name__ == "__main__":

    flag_debug = 0
    flag_index = False # if true, plot the spectrogram wrt to index; if False, wrt to Timestamps and Frequencies
    fontdict = {'fontsize':10}
    vmin, vmax = 0, 60
    vmin, vmax = None, None
    n_ticks = 10

    # we plot the raw spectrogram and two interpolated spectrograms for the following classes
    sel_classes = ["Run"]
    nsel = len(sel_classes)
    functions = {"raw spectrogram":           "none",
                 "linear interpolation":      "linear",
                 "logarithmic interpolation": "log"}

    remaining_classes = sel_classes.copy()
    index = 3204  # where to tart the search

    signal_name = "Acc_norm"
    temporal_transform = TemporalTransform([signal_name])  # we will plot the result

    while len(remaining_classes)>0:

        flag_debug = len(remaining_classes) == (nsel-1)

        data_tensor, class_tensor = DS[index]
        data_cpu = {signal:data_tensor[signal].to(torch.device('cpu')).detach().numpy() for signal in data_tensor.keys()}
        class_index = int(class_tensor)

        class_name = classes_names[class_index-1]

        prefix_title = '%s (index=%d)'% (class_name,index)



        if class_name in remaining_classes:
            remaining_classes.remove(class_name)


            temporal_signal = temporal_transform(data_cpu)[signal_name]
            nb = temporal_signal.shape[1]
            x_t = np.linspace(0, nb/fs, nb)


            plt.figure(figsize=(30,10))

            plt.subplot(2,4,1)


            plt.plot(x_t, temporal_signal[0,:])
            plt.title(prefix_title + "\nraw signal : {}".format(signal_name), fontdict)
            plt.xlabel("t (sec)")
            plt.ylabel("Acc (m/s²)")

            index_figure = 2


            # for log_power in [False]:
            for log_power in [False, True]:

                if flag_debug:
                    print('\n log_power = %s' % log_power)

                for f_name in functions :

                    if flag_debug:
                        print('\n f_name = %s' % f_name)

                    function_interpol = functions[f_name]

                    data_tensor, _ = DS[index]  # we need to recreate data because the variable is deleted
                    data_cpu = {signal:data_tensor[signal].to(torch.device('cpu')).detach().numpy() for signal in data_tensor.keys()}

                    spectrogram_transform = SpectrogramTransform([signal_name], fs, duration_window, duration_overlap, spectro_batch_size,
                                                                 function_interpol, log_power, out_size=(48, 48), flag_debug = flag_debug)

                    spectrogram_interpolated = spectrogram_transform(data_cpu)[signal_name]

                    f_interpolated = spectrogram_transform.f_interpolated
                    t_interpolated = spectrogram_transform.t_interpolated


                    error_message_dtype = "One of the spectrograms does not have the correct type: {}, log_power={}, {}. \n dtype should be float32, is actually {}".format(signal_name, str(log_power), f_name, spectrogram_interpolated.dtype)
                    assert (spectrogram_interpolated.dtype == 'float32'), error_message_dtype

                    plt.subplot(2,4,index_figure)

                    if flag_index:
                        ylabel = "f (index)"
                        xlabel = "t (index)"

                        plt.imshow(spectrogram_interpolated[0,:,:])

                    else:
                        ylabel = "f (Hz) "
                        xlabel = "t (s)"

                        t_interpolated = spectrogram_transform.t_interpolated
                        f_interpolated = spectrogram_transform.f_interpolated
                        matrix_shape = spectrogram_interpolated.shape
                        time_list = [f'{t_interpolated[i]:.0f}' for i in np.round(np.linspace(0, matrix_shape[2]-1,n_ticks)).astype(int)]
                        freq_list = [f'{f_interpolated[i]:.1f}' for i in np.round(np.linspace(0, matrix_shape[1]-1,n_ticks)).astype(int)]

                        plt.xticks(np.linspace(0, matrix_shape[2]-1, n_ticks), time_list)
                        plt.yticks(np.linspace(0, matrix_shape[1]-1, n_ticks), freq_list)

                        plt.imshow(spectrogram_interpolated[0,:,:])

                    plt.ylabel(ylabel)
                    plt.xlabel(xlabel)
                    plt.colorbar()

                    index_figure += 1

                    log_power_text = 'log power' if log_power==True else 'power'

                    plt.title("{} of {}".format( log_power_text, f_name), fontdict = {'fontsize':10})

                index_figure += 1 # for the vertical alignment
        index +=1
    plt.show()

