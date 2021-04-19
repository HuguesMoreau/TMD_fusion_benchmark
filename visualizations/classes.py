"""
Author Hugues

Some basic visualizations on the data:
- mode change count: as we plan to return only one mode per sample, we need to
    make sure we do not misclassify too many points
- class distribution: shows why we split using (val:3310, train:13000) instead
    of (train:13000, val:3310)

This script uses the pickles created by reorder.py, and the Label.txt file

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

"""

if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")


import numpy as np
import matplotlib.pyplot as plt
import time

from param import classes_names
from preprocess.Datasets import SignalsDataSet


# parameters of the bar graph
stride = 50             # number of instances
window_size = 1500      # number of instances
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0.9, 0.9, 0], [0, 0, 0], [0.5, 0, 0]]  # color per class


n_classes = len(classes_names)

start_time = time.time()



if __name__ == "__main__":
    #%% class histograms
    # opening files
    modes = ["train", "val", "test"]
    splits = ["unbalanced", "balanced", "shuffle"]


    label_arrays = {}

    for mode in modes:
        labels_this_mode = {}
        for split in splits:
            dataset = SignalsDataSet(mode, split, comp_preprocess_first=False, transform=None)
            label_mat = dataset.labels

            segment_size = label_mat.shape[1]
            # to assign a single class to each segment, we take the label at the middle of the segment
            labels_this_mode[split] = label_mat[:,segment_size//2].astype(int)   # (n_mode_samples,)

        label_arrays[mode] = labels_this_mode

    del dataset


    n_classes = max(label_arrays["train"]["shuffle"])
    #%%
    titles = {'balanced':'val/train', 'unbalanced':'train/val', 'shuffle':'challenge'}
    plt.figure(figsize=(10,10))
    for i_split, split in enumerate(splits):
        for i_mode, mode in enumerate(["train", "val", "test"]): # note: the competitors did not have access to the test distribution
            print(mode, split, "\t", np.histogram(label_arrays[mode][split], bins=8)[0])

            ax = plt.subplot(3,3, 3*i_mode + i_split +1)
            plt.grid('on', zorder=-5)
            #fig, ax = plt.subplots()

            bins = np.arange(0, 1+n_classes)+0.5  # [0.5, 1.5, 2.5, ..., 9.5]
            plt.hist(label_arrays[mode][split], bins=bins, rwidth=0.95, density=True, zorder=5)
                # labels range from 1 to 8, included
            plt.xticks(1+np.arange(n_classes), classes_names, fontsize=6)
            ax.set_ylim(0,0.25)
            yticks =  np.round(np.linspace(0, 0.25, 6), 3)
            plt.yticks(yticks, yticks, fontsize=6)  # resize the ticks

            plt.title(f"{mode} data, {titles[split]} split, {label_arrays[mode][split].shape[0]} samples", fontsize=8)

        plt.plot()

    print("train+val\t", np.histogram(label_arrays["train"]["shuffle"], bins=8)[0] + np.histogram(label_arrays["val"]["shuffle"], bins=8)[0])

    #%%
    # number of changes per segment
    # or why we do not consider returning one segment per data point :
    # we look at each segment, and notice the number of points we would
    # misclassify if we were to assign a single class to the whole segment
    # would be quite small


    train_dataset = SignalsDataSet("train", "unbalanced", comp_preprocess_first=False, transform=None)
    val_dataset   = SignalsDataSet("val",   "unbalanced", comp_preprocess_first=False, transform=None)

    label_segments_list = list(train_dataset.labels) + list(val_dataset.labels)

    num_segments = len(label_segments_list)
    num_points_total = num_segments * segment_size

    num_segments_with_change = 0
    num_misclassified_points = 0

    for label_segment in label_segments_list:
        change_in_segment = (label_segment.max() != label_segment.min())

        if change_in_segment :
            num_segments_with_change += 1

            chosen_class = label_segment[segment_size//2]
            errors = (label_segment != chosen_class)
            num_misclassified_points += np.sum(errors)

    print('\n\nOut of {} segments in total, {} have a mode change ({:.3f}%)'.format(num_segments, num_segments_with_change, 100*num_segments_with_change/num_segments))
    print('Proportion of points we are about to misclassify no matter what: {:.3f} % \n\n'.format(100*num_misclassified_points/num_points_total))


    #%%
    # ------------------ Proportions on the train folder  ---------------------

    # To understand why we need to take the first part of the samples as val,
    # and the last part as train, we plot the distribution of classes,
    # computed on a moving window. We will see that there are almost no car segments in
    # the last 2,500 segments of the file. The beginning of the file, on the other
    # hand, is not that imbalanced.

    train_labels_sorted = [ label[label.shape[0]//2] for label in label_segments_list ]

        # this is more simple than aligning all 6,000 labels from each of the 16,000 segments


    # ---------------- Plotting a bar chart  --------------------
    # inspired from   https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py



    plt.subplots(figsize=(15, 8))
    for i in range(0, len(train_labels_sorted), stride):
        prev_i = max(i-window_size, 0)
        next_i = min(i+stride+window_size, len(train_labels_sorted))

        histogram = np.histogram(train_labels_sorted[prev_i:next_i], bins=n_classes)[0]
        proportions = histogram/np.sum(histogram)
        cum_proportions = np.cumsum(proportions)                                         # from class 1 to class  n  (included)
        cum_proportions = np.concatenate([np.array([0]), cum_proportions[:-1]], axis=0)  # from class 0 to class n-1 (included)

        x = np.array([i]*n_classes)
        width = np.array([stride]*n_classes)
        bars = plt.bar(x, proportions, width=width, bottom=cum_proportions, color=colors)
            # we only keep the last bar created, for the legend

#    plt.title('moving average of the class distribution (window size=%d)'%window_size, fontsize='x-large')

    plt.xlabel("sample index")#, fontsize='x-large')
    plt.ylabel("proportion")#, fontsize='x-large')
    plt.legend(bars, classes_names)
    plt.show()



