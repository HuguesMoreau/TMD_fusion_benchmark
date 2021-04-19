#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains diverse preprocessing functions (mostly norms ans spectrograms),
and basic tests and visualizations.
If you are to work with any IPython console (ex: with Jupyter or spyder), is is advised
to launch a '%matplotlib qt' ,to get clean widows
"""


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 8
from math import ceil, floor

import numpy as np
import torch

from preprocess.Datasets   import SignalsDataSet
from preprocess.reorder import classes_names, segment_size
from preprocess.transforms import FFTTransform


n_classes = len(classes_names)
xscale = 'linear'      # log, linear
yscale = 'log'

plot_std = False

signals_list = ['Acc_norm', 'Gyr_y', 'Mag_norm', 'Ori_w', 'Ori_norm', 'Pressure']
#signals_list = ['Gra_norm', 'Gra_x', 'Gra_y', 'Gra_z']


#%% FFT computation


signal_spectrum = {}

for signal in signals_list:  # we process one signal at a time because comuting all of them at once might be too much
    transform = FFTTransform([signal])

    DS_train = SignalsDataSet(mode='train', split='balanced', comp_preprocess_first=True, transform=transform)
    DS_val =   SignalsDataSet(mode='val',   split='balanced', comp_preprocess_first=True, transform=transform)
    spectrum = torch.cat([DS_train.data[signal], DS_val.data[signal]], dim=0)

    Label = torch.cat([DS_train.labels[:,segment_size//2], DS_val.labels[:,segment_size//2]], dim=0)
    Label = Label.detach().cpu().numpy().astype('int') -1
    # Labels is now a 1-dimensional array (containing integers frm 0 to 7, included)

    del DS_train, DS_val

    spectrum = spectrum.detach().cpu().numpy()
    half_spectrum = spectrum[:,segment_size//2:]

    signal_spectrum[signal] = half_spectrum
    del spectrum, half_spectrum




"""

transform = FFTTransform(signals_list)

# We will need this for the tests
DS_train = SignalsDataSet(mode='train', split='balanced', comp_preprocess_first=True, transform=transform)
DS_val =   SignalsDataSet(mode='val',   split='balanced', comp_preprocess_first=True, transform=transform)

# FFT computation


signal_spectrum = {}
for signal in DS_train.signals:
    spectrum = torch.cat([DS_train.data[signal], DS_val.data[signal]], dim=0)
    del DS_train.data[signal], DS_val.data[signal]
    spectrum = spectrum.detach().cpu().numpy()
    half_spectrum = spectrum[:,segment_size//2:]

    signal_spectrum[signal] = half_spectrum
    del spectrum, half_spectrum

Label = torch.cat([DS_train.labels[:,segment_size//2], DS_val.labels[:,segment_size//2]], dim=0)
Label = Label.detach().cpu().numpy().astype('int') -1
# Labels is now a 1-dimensional array (containing integers frm 0 to 7, included)

"""


#%% Display average power
n_signals = len(signals_list)
n_col = floor(np.sqrt(n_signals))
n_lines = ceil(n_signals/n_col)


plt.figure(figsize=(n_lines*4,n_col*4))

x_f = np.linspace(0,50, 3000)
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0.9, 0.9, 0], [0, 0, 0], [0.3, 0, 0]]




for i_signal, signal in enumerate(signals_list):
    ax = plt.subplot(n_lines,n_col, i_signal+1)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plotted_values = 'mean ± std' if plot_std else 'mean'
    plt.title(f"{plotted_values} power spectrum of {signal}")
#    plt.legend(classes_names)
    if i_signal //n_col == n_lines-1:  plt.xlabel('f (Hz)')  # only on the bottom graphs
    ymin, ymax = np.inf, -np.inf

    for i_class, class_name in enumerate(classes_names):

        mask = (Label == i_class)
        assert ((Label[mask].max() == Label[mask].min()) and (int(Label[mask].min()) == i_class))
                # all labels are the same and equal to i_class

        selected_samples = signal_spectrum[signal][mask,:]

        average_power = np.mean(selected_samples, axis=0)
        std_power = np.std(selected_samples, axis=0)


        plt.plot(x_f[1:], average_power[1:], color=colors[i_class])
        ymin = min(ymin, average_power[1:].min())  # the mean components, average_power[0], was set to zero
        ymax = max(ymax, average_power.max())

        if plot_std : plt.fill_between(x_f[1:], average_power[1:]+std_power[1:], average_power[1:]-std_power[1:], color=colors[i_class]+[0.15])  # added alpha=0.1


    plt.legend(classes_names, ncol=2, fontsize=6)
    margin = 1.2 if yscale== 'linear' else 2
    ax.set_ylim(ymin / margin,  ymax * margin)
    plt.grid('on')


plt.show()
#plt.tight_layout()







