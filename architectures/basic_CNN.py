"""
Author Hugues

The CNN Defined here allows to implement all kinds o early fusion, and feature
fusion
"""


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")


import numpy as np
from math import floor
import torch
import torch.nn as nn

from architectures.base_network import Network



def next_shape(input_shape, conv_layer, maxpool_layer):
    """
    Compute the output shape of a single layer, using the input shape and the layer itself
    This fucion help us assess automatically the input size of the first linear
    layer after the flatten step

    Parameters
    ----------
    - input_shape (triple of ints, (channel, freq, time))
    - conv_layer: the torch.nn.Conv2d or torch.nn.Conv1d object
    - maxpool_layer: the torch.nn.MaxPool2d oor torch.nn.MaxPool1d object

    Returns: A triple (channel, freq, time)
    """

    padding = conv_layer.padding           # 1-tuple or 2-tuple (depending on the convolution type)
    kernel_size = conv_layer.kernel_size   # 1-tuple or 2-tuple
    stride = maxpool_layer.stride          # either integer or 2-tuple

    output_shape = list(input_shape) # allows item assignment
    output_shape[0] = conv_layer.out_channels

    if len(input_shape) == 2: #1D signal
        output_shape[1] += 2*padding[0] - (kernel_size[0] -1)
        output_shape[1] = floor(output_shape[1]/stride)

    else: #2D signal
        output_shape[1] += 2*padding[0] - (kernel_size[0] -1)
        output_shape[2] += 2*padding[1] - (kernel_size[1] -1)

        output_shape[1] = floor(output_shape[1]/stride[0])
        output_shape[2] = floor(output_shape[2]/stride[1])

    return tuple(output_shape)




#%%
class CNN(Network):
    """
    A convolutional network that allows basic operations : early to late fusion,
    using depth-wise, time-wise, or channel-wise concatenation.
    Depending on the input_shape parameter, the convolutions are either 1D or 2D
    """

    def __init__(self, input_shape, n_branches, fusion):
        """
        Parameters
        ----------
        - input_shape: either a couple (channel, time); or a triple (channel, freq, time)
            the input shape for 1 branch
        - n_branches (int): How many convolution branches to keep. Must be at
            least 1. In case of depthwise concatenation, the different channels
            going to a single branch must be contiguous.
            n_branches = 1 for classic network; >1 for feature fusion
        - fusion (string, one of 'start', 'conv0', 'conv1', 'conv2'):
            where to perform the fusion. Each time, the fusion is made
            right after the mentionned layer (except for 'start')
        """

        super(CNN, self).__init__()
        data_type = "1D" if len(input_shape) == 2 else "2D"
        n_input_channels = input_shape[0] * n_branches

        if fusion == 'start': n_branches = 1

        if data_type == "1D":
            ConvLayer = nn.Conv1d
            self.mp = nn.MaxPool1d(2)
        if data_type == "2D":
            ConvLayer = nn.Conv2d
            self.mp = nn.MaxPool2d((2,2))

        self.conv0 = ConvLayer( n_input_channels, n_branches*16, 3, padding=1, groups=n_branches)
        output_shape = next_shape(input_shape, self.conv0, self.mp)
        if fusion == 'conv0': n_branches = 1
        self.relu_conv0 = nn.ReLU()   # we define one RELU function per layer make overriding easier

        self.conv1 = ConvLayer(n_branches*16,     n_branches*32, 3, padding=1, groups=n_branches)
        output_shape = next_shape(output_shape, self.conv1, self.mp)
        if fusion == 'conv1': n_branches = 1
        self.relu_conv1 = nn.ReLU()

        self.conv2 = ConvLayer(n_branches*32,     n_branches*64, 3, padding=0, groups=n_branches)
        output_shape = next_shape(output_shape, self.conv2, self.mp)
        # no need to set number_branches=1 here
        self.relu_conv2 = nn.ReLU()

        self.relu_FC0 = nn.ReLU()


        FC_shape = np.product(output_shape)
        self.dropout0 = nn.Dropout(0.25)
        self.FC0 = nn.Linear(FC_shape,128)
        self.dropout1 = nn.Dropout(0.5)
        self.FC1 = nn.Linear(128, 8)  # 8=nb of classes

        self.softmax = nn.Softmax(dim=1)



    def forward(self, X):
        """
        Input :
            X : an input tensor. Its shape should be equal to
                (B, input_shape[0] * n_branches, input_shape[1], input_shape[2])
        Output :
            scores : a (B, 8) tensor
        """
        X = self.conv0(X)
        X = self.mp(self.relu_conv0(X))
        X = self.conv1(X)
        X = self.mp(self.relu_conv1(X))
        X = self.conv2(X)
        X = self.mp(self.relu_conv2(X))
        X = X.view(X.shape[0],-1)
        X = self.dropout0(X)

        X = self.relu_FC0(self.FC0(X))

        X = self.dropout1(X)
        X = self.FC1(X)

        return X






#%%
if __name__ == "__main__":


    import torch.utils.data
    from preprocess import Datasets
    from preprocess.transforms import SpectrogramTransform
    from preprocess.fusion import ConcatCollate
    from architectures.base_network import device


    from param import fs, duration_window, duration_overlap, spectro_batch_size
    # spectrogramm network
    transform = SpectrogramTransform(["Acc_norm"], fs, duration_window, duration_overlap, spectro_batch_size,
                                                 interpolation='log', log_power=True, out_size=(48,48))


    collate_fn = ConcatCollate("time")   # concatenate signals and merge into a Batch

    train_dataset = Datasets.SignalsDataSet(mode='train', split='balanced', comp_preprocess_first=True, transform=transform)
    val_dataset =   Datasets.SignalsDataSet(mode='val',   split='balanced', comp_preprocess_first=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, collate_fn=collate_fn, shuffle=False)


    model = CNN(input_shape=(1, 48,48), fusion='start', n_branches=1)
    model.to(device=device)

    model.train_process(train_dataloader, val_dataloader, maxepochs=50)
    model.plot_learning_curves()
    model.optim.zero_grad()
    model.plot_confusion_matrix(val_dataloader)


