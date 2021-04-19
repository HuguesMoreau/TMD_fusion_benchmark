"""
Author Hugues

CNN for bottleneck filters
"""


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn

from architectures.base_network import Network
from architectures.basic_CNN import next_shape


class bottleneckCNN(Network):
    """
    A convolutional network with a bottleneck layer at the start.
    """

    def __init__(self, input_shape):
        """
        Parameters
        ----------
        - input_shape: either a couple (channel, time); or a triple (channel, freq, time)
        """

        super(bottleneckCNN, self).__init__()
        data_type = "1D" if len(input_shape) == 2 else "2D"

        if data_type == "1D":
            ConvLayer = nn.Conv1d
            self.mp = nn.MaxPool1d(2)
        if data_type == "2D":
            ConvLayer = nn.Conv2d
            self.mp = nn.MaxPool2d((2,2))

        self.relu = nn.ReLU()

        self.conv_sensors = ConvLayer(input_shape[0], 1, 1, padding=1)
        output_shape = (1,) + input_shape[1:]
            # same as input_shape, but the number of channels is set to 1

        self.conv0 = ConvLayer( 1, 16, 3, padding=1)
        output_shape = next_shape(output_shape, self.conv0, self.mp)

        self.conv1 = ConvLayer(16, 32, 3, padding=1)
        output_shape = next_shape(output_shape, self.conv1, self.mp)

        self.conv2 = ConvLayer(32, 64, 3, padding=0)
        output_shape = next_shape(output_shape, self.conv2, self.mp)

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

        X = self.conv_sensors(X)
        X = self.relu(X)  # no maxpool for this layer

        X = self.conv0(X)
        X = self.mp(self.relu(X))
        X = self.conv1(X)
        X = self.mp(self.relu(X)) # here
        X = self.conv2(X)
        X = self.mp(self.relu(X))
        X = X.view(X.shape[0],-1)
        X = self.dropout0(X)

        X = self.relu(self.FC0(X))

        X = self.dropout1(X)
        X = self.FC1(X)

        return X



    def get_weights(self, signals_list, dataloader=None):
        """
        Paramters
        ---------
        signals_list: a list of string, used to make the correspondance
            between the weights and the signals
        dataloader(optional): an argument to mimick the signature of
            AttentionCNN.get_weights.
            It is systematically ignored


        Returns
        -------
        weights: a dictionnary of floats
        """
        coefficients = {}
        for i, signal in enumerate(signals_list):
            coefficients[signal] = self.conv_sensors.weight.data[0,i,0,0].item()
        return coefficients

#%%
if __name__ == "__main__":
    from preprocess import Datasets
    import torch.utils.data
    from preprocess.transforms import SpectrogramTransform
    from preprocess.fusion import ConcatCollate
    from param import fs, duration_window, duration_overlap, spectro_batch_size, device

    # test

    transform = SpectrogramTransform(["Acc_norm", "Gyr_y"], fs, duration_window, duration_overlap, spectro_batch_size,
                                                 interpolation='log', log_power=True, out_size=(48,48))
#    transform = TemporalTransform(["Acc_norm"])

    collate_fn = ConcatCollate("depth")   # concatenate signals and merge into a Batch
    try :  # do not reload the datasets if they already exist
        train_dataset
        val_dataset


    except NameError:
        train_dataset = Datasets.SignalsDataSet(mode='train', split='balanced', comp_preprocess_first=True, transform=transform)
        val_dataset =   Datasets.SignalsDataSet(mode='val',   split='balanced', comp_preprocess_first=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, collate_fn=collate_fn, shuffle=False)

    model = bottleneckCNN(input_shape=(2, 48, 48))
    model.to(device)

    model.train_process(train_dataloader, val_dataloader, maxepochs=50)
    torch.save(model, 'model.py')
