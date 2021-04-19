"""
Author Hugues

Contains the utility toload the ordered data, split the train and validation
sets, and apply the reprocessing function

There are three ways to split the data:
    - 'shuffle': choose 13,000 samples at random, they will go in the training set,
        the other 3,310 go in the validation set. This is equivalent to doing
        like in the challenge, where most participant did not take the order
        into account. This also leads to overfitting, as fragments of a single
        trajectory can go in both sets, which means there is a possible
        contamination
    - 'unbalanced': the first 13,000 samples of the dataset (sorted chronologically)
        go in the training set, the other 3,310 go in the validation set.
        This is not ideal, for the sets now have different distributions.
        See visualizations/classes.py to understand the cause of the unbalance
    - 'balanced': the last 13,000 13,000 samples of the dataset (sorted chronologically)
        dataset go in the training set, the first 3,310 go in the validation set.
        This is the best way, for it produces sets with similar disributions

Note that this separation only applies to the training and validation sets,
the test set is kept the same as in th official challenge

"""

if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")


import torch.utils.data
from timeit import default_timer as timer


import numpy as np

from pathlib import Path
from param import data_path, device
from preprocess.reorder import base_signals, segment_size

import pickle


len_train = 13000 # 13,000 elements in the train set





#%%

class SignalsDataSet(torch.utils.data.Dataset):
    def __init__(self, mode, split, comp_preprocess_first, transform=None, continuous_labels=False):
        """
        Create a dataset for one mode (train, test, val), and one position (hand, pocket, etc.).
        Depending on the comp_preprocess_first option, two behaviours are available:
        if comp_preprocess_first = False: the Dataset object stores raw data (a dict of
            (_, 6000) numpy arrays) on the CPU, it applies the transform function
            and sends the data to the GPU each time it is asked one element
            (in the __getitem__ method).
        if comp_preprocess_first = True: the Dataset object applies the transform method
            and sends the data to the GPU once (when the data is loaded),
            and stores the result

        comp_preprocess_first = False makes the whole computation longer, especially
        with spectrograms (as the same sample is applied the transform function
        repeatedly), but is useful when the preprocessed data does not fit in memory
        (ex: with raw, uninterpolated spectrograms)


        Parameters
        ----------
        - mode (string): either 'train', 'val', or 'test'
        - split (string): either 'shuffle', 'unbalanced', or 'balanced'
        - comp_preprocess_first (bool): see above
        - transform (function) : the function to apply to all samples (default: identity)
            It must take 2d arrays (shape: (_,6000)) as inputs, and return
            a numpy array as an output.
        - continuous_labels (bool): if False (default), there is only one label
            per segment (most common case). If True, one label per data point
            (6000 label values, only for the transformer)

        Returns
        -------
        a dataset object, containing a dict of numpy arrays
        """
        self.mode = mode
        self.split = split
        self.comp_preprocess_first = comp_preprocess_first
        self.transform = transform
        self.continuous_labels = continuous_labels

        directory = "test" if mode == "test" else "train"
        file_path = Path(directory) / Path("ordered_data.pickle")
        self.path = file_path # useful for debugging

        complete_path = data_path / Path(file_path)
        print("\nDataset creation\n mode: {}, split: {}, comp_preprocess_first: {}".format(mode, split, comp_preprocess_first))
        print(" loading '{}'... ".format(complete_path), end='')


        start = timer()
        with open(complete_path, "rb") as f:
            dict_data = pickle.load(f)
        end = timer()
        count = (end - start)
        print('load Data in %.2f sec' % count)

        len_file = dict_data['Label'].shape[0]
        len_val = len_file - len_train

        # ===================================================================
        #                         train/val split
        # =====================================================================


        if mode in ["train", "val"]:
            train_order = dict_data['order']
            del dict_data['order']
            start = timer()
            if split == "shuffle":
                train_index = train_order[:len_train] -1
                val_index   = train_order[len_train:] -1

            elif split == "unbalanced":
                train_index = range(0,         len_train)
                val_index   = range(len_train, len_file)

            elif split == "balanced":
                train_index = range(len_val, len_file)
                val_index   = range(0,       len_val)

            if mode == "train":
                chosen_index = train_index
            else: # mode == "val":
                chosen_index = val_index

            dict_data = {signal:dict_data[signal][chosen_index,:] for signal in dict_data.keys()}

            end = timer()
            count = (end - start)
            print('Split Train/Val in %.2f sec' % count)

        else: # test mode
            del dict_data['order']  # the order is not taken into account
                            # as we only use the test set for evaluation

        # =====================================================================
        #                    Apply preprocessing
        # =====================================================================

        if comp_preprocess_first == False:
            self.labels = dict_data["Label"]
            del dict_data["Label"]

            self.signals = base_signals
            if self.transform != None:
                print("preprocessing function to be applied later:", self.transform)

        else:
            start = timer()

            self.labels = torch.tensor(dict_data["Label"], dtype=torch.long, device=device) # a Pytorch tensor on GPU
            del dict_data["Label"]


            print('before preprocessing:', end=' ')
            print({signal:dict_data[signal].shape for signal in dict_data})


            if self.transform != None:
                print("starting to apply preprocessing function:", self.transform)
                dict_data = transform(dict_data)

            dict_data = {signal_name:torch.tensor(dict_data[signal_name], dtype=torch.float32, device=device) for signal_name in dict_data.keys()}

            self.signals = list(dict_data.keys())
            print('after preprocessing: ', {signal:dict_data[signal].shape for signal in dict_data})

            end = timer()
            count = (end - start)
            print('Preprocessing in %.2f sec' % count)



        self.n_signals = len(self.signals)
        matrix_example = dict_data[self.signals[0]] # to get the number of samples, we take the first matrix we have
        self.n_points = matrix_example.shape[1] # number of points per sample
        self.len = matrix_example.shape[0]     # number of samples

        if comp_preprocess_first == False: assert (self.n_points == segment_size), "Inconsistency in the number of points per sample. Expected {}, got {}".format(self.n_points, segment_size)
             # reminder : segment_size comes from reorder.py
        print("{} signals ({}), with {} samples per signal".format(self.n_signals, list(dict_data.keys()), self.len), end="\n\n")

        self.data = {signal:dict_data[signal] for signal in self.signals}


    def __len__(self):
        return self.len

    def __getitem__(self,i):
        """
        AV: called when DS[i] , DS being an instance of SignalsDataSet
        Parameters
        ----------
        i is an integer between -self.len and self.len-1 (inclusive)


        Returns
        -------
        (selected_data, label)
        - selected_data is a dictionnary with signals as keys, and
            torch.FloatTensor as values (shape: (1,6000) without preprocessing)
        - label is a torch.LongTensor with shape (1,)    (if self.continuous_labels == False)
                or a torch.LongTensor with shape (6000,) (otherwise)

        """

        if (i < -self.len or i >= self.len):
            raise IndexError("Incorrect index in '{}' dataset : {} (length of the dataset is {})".format(self.path, i, self.len))
        i = i%self.len # allows to use negative index

        label_list = self.labels[i,:] # a series of 6000 labels

        if self.continuous_labels == False : # we want one label per segment
            label = label_list[label_list.shape[0]//2] # arbitrary choice: choose the label
                                                       # in the middle of the segment
        else :
            label = np.array(label_list, dtype=np.float32)

        if self.comp_preprocess_first == False:
            if self.transform != None:
                selected_data = self.transform({signal:self.data[signal][i:i+1,:] for signal in self.signals})
            else :
                selected_data = {signal:self.data[signal][i:i+1,:] for signal in self.signals}
            # In both cases, send the data to the GPU
            selected_data_gpu = {}
            for signal_name in selected_data.keys():
                signal_data = selected_data[signal_name]

                if type(signal_data) == np.ndarray:
                    signal_data = torch.tensor(signal_data, dtype=torch.float32, device=device)
                else:
                    signal_data = signal_data.to(device=device)

                selected_data_gpu[signal_name] = signal_data

            # In both cases, send the data to the GPU
            selected_data = {signal_name:torch.tensor(selected_data[signal_name], dtype=torch.float32, device=device)
                            for signal_name in selected_data.keys()}

            if (type(label) == np.ndarray) or (type(label) == np.float32):
                label = torch.tensor(label, dtype=torch.long, device=device)
            else:
                label = label.to(dtype=torch.long, device=device)

        else : # the data is already on the GPU
            selected_data_gpu = {signal:self.data[signal][i:i+1,:] for signal in self.signals}

        return (selected_data_gpu, label)






#%%
if __name__ == "__main__":
    """
    Print some samples from the original txt file, along with the ame samples
    from two datasets (corresponding to different slits)
    """

    import time

    # --------------  shuffled split  --------------
    print('\n\n\n')

    start_time = time.time()
    DS = SignalsDataSet(mode="train", split="shuffle", comp_preprocess_first=False, transform=None)
    loading_time = time.time() - start_time
    print("data was successfully loaded in {:.3f} seconds".format(loading_time))

    # Compare the first few elements of each file
    # with shuffle mode, the order should be the same
    signal = "Gyr_z"
    filepath = data_path / Path("train/" + signal + ".txt")
    print("\n Comparisons between files ({})".format(filepath))
    with open(filepath, "r") as f_signal:
        for i in range(10):
            data = next(f_signal)

            print("original txt file:         ", data[:30] + "..." + data[-30:-1])  # we do not print the last '\n'
            print("Dataset object    ", DS[i][0][signal])
            print('\n')

    # --------------  unbalanced split  --------------
    start_time = time.time()
    DS = SignalsDataSet(mode="train", split="unbalanced", comp_preprocess_first=False, transform=None)
    loading_time = time.time() - start_time
    print("data was successfully loaded in {:.3f} seconds".format(loading_time))

    order_list = np.loadtxt(data_path / Path("train/train_order.txt")).astype(int)

    # Compare the first few elements of each file
    signal = "Gyr_z"
    filepath = data_path / Path("train/" + signal + ".txt")
    print("\n Comparisons between files ({})".format(filepath))
    with open(filepath, "r") as f_signal:
        for i in range(10):
            order = order_list[i]
            data = next(f_signal)
            i_dataset = order -1

            if order-1 < len_train:
                print(signal)
                print("original txt file:         ", data[:30] + "..." + data[-30:-1])  # we do not print the last '\n'
                print("Dataset object    ", DS[i_dataset][0][signal])
                print('\n')


    # --------------  balanced split  --------------
    print('\n\n\n')

    start_time = time.time()
    DS = SignalsDataSet(mode="train", split="balanced", comp_preprocess_first=False, transform=None, continuous_labels=False)
    loading_time = time.time() - start_time
    print("data was successfully loaded in {:.3f} seconds".format(loading_time))

    order_list = np.loadtxt(data_path / Path("train/train_order.txt")).astype(int)
    len_val = len(order_list) - len_train

    # Compare the first few elements of each file
    signal = "Gyr_z"
    filepath = data_path / Path("train/" + signal + ".txt")
    print("\n Comparisons betweeen files ({})".format(filepath))
    with open(filepath, "r") as f_signal:
        for i in range(10):
            order = order_list[i]
            data = next(f_signal)
            i_dataset = order -1 -len_val

            if order-1 > len_val:
                print("original txt file:         ", data[:30] + "..." + data[-30:-1])  # we do not print the last '\n'
                print("Dataset object    ", DS[i_dataset][0][signal])
                print('\n')

