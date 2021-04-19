#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author Hugues

This scripts selects the signals given by chosen_signals, and reorders the data
according to the train_order.txt and test_order_txt files.
It then saves a .pickle in [data_path]/{train|test}/ordered_data.pickle

Expected file tree :
[data_path]
    train
        Acc_x.txt
        Acc_y.txt
        Acc_z.txt
        Gyr_x.txt
        etc.
    test
        Acc_x.txt
        etc.


The saved data is a dictionnary with keys such as "Acc_y", and numpy arrays as *
values (shapes: (N_samples, 6000)). We use .pickle instead of .txt files because
loading the dataset is ~40 times faster.


Vocaulary: in this project, a sensor can be "Acc" or "Gyr", and a signal can be
"Acc_x", or "Acc_y".


Note: The arrays are stored as simple precision digits (float32) instead of double
precision (float64). This has no consequence, because:
1) Pytorch will do this conversion when it sends the data to the GPU and
2) the original .txt files contain values with 7 decimals, which is close to the simple
    precision: 23 bits mantissa -> 2**(-23) = 1.19e-7 relative precision. Given
    that the values we chose to keep have an absolute value bounded by ~100,
    we might lose 1 or 2 significant digits when the nominal value is 10 or 100.
    no big deal

"""

if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")


import numpy as np

import pickle
from pathlib import Path
from param import data_path


all_signal_filenames = [signal+"_"+coord+'.txt' for signal in ["Acc","Gra","LAcc","Gyr","Mag","Ori"]
                                                    for coord in ["x","y","z"]] \
                            + ["Ori_w.txt", "Pressure.txt" ]


classes_names = ["Still","Walk","Run","Bike","Car","Bus","Train","Subway"]
# cf the last page of    Wang, Lin, Hristijan Gjoreskia, Kazuya Murao, Tsuyoshi Okita, and Daniel Roggen. « Summary of the Sussex-Huawei Locomotion-Transportation Recognition Challenge »
# https://doi.org/10.1145/3267305.3267519.


debug = 0
    # if debug == 1, apply the same treatment to train_order.txt instead
    # of real data (Acc_x, Gyr_y, etc.). It will allow us to check everything
    # is here, and that the segments split according to their order.
    # be careful, debug == 1 *overwrites* the data on the disk

if debug == 0 :
    base_signals = [signal+"_"+coord for signal in ["Acc","Gra","LAcc","Gyr","Mag","Ori"] for coord in ["x","y","z"]] + ["Ori_w", "Pressure" ]
    # you may change here depending on what signals you want to keep
elif debug == 1 :
    base_signals = ["train_order"]
    print("=============================== Debug mode ===============================")



segment_size = 6000 if debug==0 else 1
    # the number of points in a sample. In real SHL 2018 samples, there are 6000 points,
    # but train_order only yields one integer per line



import time
start_time = time.time()

if __name__ == "__main__":
    for mode in ["train", "test"]:
        chosen_signals = base_signals + ["Label"] if debug == 0 else base_signals

        print("\n ---- {} mode ----".format(mode))
        print('chosen signals: ', chosen_signals)

        mode_path = data_path / Path(mode)

        """Principle: we load all the data into a dictionnary of numpy arrays,
        before writing the values in a pickle. We look at each of the samples,
        and append the sample to the line corresponding to its order"""

        if mode == "train":
            order_array = np.loadtxt(mode_path/Path("train_order.txt")).astype(int)
        elif mode == "test":
            order_array = np.loadtxt(mode_path/Path("test_order.txt")).astype(int)

        # to compute the number of segments, we need one file (all files are assumed to have the same number of segments)
        example_file = mode_path / Path(chosen_signals[0]+".txt")
        with open(example_file) as f:
            len_file = sum(1 for row in f)
        file_locations = {signal:mode_path/Path(signal+".txt") for signal in chosen_signals}

        data_dict = {signal: np.zeros((len_file, segment_size), dtype='float32') for signal in chosen_signals }

        for signal in chosen_signals:
            print("\n", signal)

            print("loading '%s'... "%file_locations[signal], end='')
            original_data = np.loadtxt(file_locations[signal])
            if debug : # in debug mode, only the order is loaded, which has only one dimension
                original_data = original_data[:,None]  # this is why we add another
            print('Done')

            len_this_file, segment_size = original_data.shape
            print("%d samples, each sample having %d points"%(len_file, segment_size))
            # sanity check : we stop if the number of lines in one file differs from the rest
            assert (len_this_file == len_file), "Two files contain a different number of segments"

            for line_index in range(len_file):
                order = order_array[line_index] -1
                data_dict[signal][order,:] = original_data[line_index,:]

            # end for line in file
        # end for signal
        data_dict['order'] = order_array


        # sanity check: look at the mode change between one segment and the following
        if debug==0:
            Labels = data_dict["Label"]
            n_changes = (Labels[:-1,-1] != Labels[1:, 0]).sum()
            print("""Reordering verification: \nthere are {} modes changes between one segment and the following
                  (should be small compared to the {} segments in total)""".format(n_changes, len_file))



        # saving the values
        filepath = mode_path / Path("ordered_data.pickle")
        with open(filepath, "wb") as f:
            pickle.dump(data_dict, f)




        #%%    Verification using train_order.txt
        if debug and mode == "train":
            del data_dict

            # test 1 : check everything is at the good location
            filepath = mode_path / Path("ordered_data.pickle")
            with open(filepath, "rb") as f:
                data_dict = pickle.load(f)

            print("the data has been saved and reloaded")


            # test 2 : check everything is here
            order_values = data_dict["train_order"]
            order_values =  [int(o) for o in order_values]
            sorted_order_values = sorted(order_values)
            expected_list = [int(o)+1 for o in range(len_file)] # keep in mind the elements of order_values are between 1 and n, included
            if sorted_order_values == expected_list :
                print("all values are here")
            else :
                # either some values are missing, or we got some unexpected values (or both)
                missing_values    = [o for o in expected_list if o not in order_values]
                print("missing order values: ", missing_values)
                unexpected_values = [o for o in order_values  if o not in expected_list]
                print("some order values are not expected: ", unexpected_values)

            # test 3: check the data is sorted
            if order_values == sorted(order_values):
                print("the reordering is correct")
            else:
                print("there has peen a problem in the reordering. The order values should be sorted, they are:")
                print([int(o) for o in order_values])

            break # do not go to test mode

    #end for mode
    print('\n processing time : %.3f min'%((time.time() - start_time)/60))




