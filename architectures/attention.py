"""
Author Hugues

This file is used to implement differents sorts of attention.
Currently, two types are implemented :

"selective_fusion":
    Chen, Changhao, Stefano Rosa, Yishu Miao, Chris Xiaoxuan Lu, Wei Wu, Andrew Markham, Niki Trigoni.
     « Selective Sensor Fusion for Neural Visual-Inertial Odometry ».
    In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10534‑43. Long Beach, CA, USA: IEEE, 2019.
    https://doi.org/10.1109/CVPR.2019.01079.


"classic":
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html?highlight=attention
"""



if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn

from param import classes_names
from architectures.basic_CNN import CNN




#%%

def batch_softmax(X):
    """
    computes a softmax along all dimensions of X except the first one
    It is equivalent to computing a softmax on all elements of X[0,...] then
    repeat for X[1,...], for X[2,...], etc.

    Parameters
    X (torch tensor with dimension > 1)

    Returns
    Y, a tensor with same shape as X, which elements are in [0,1]
    """
    exp_X = torch.exp(X)

    sum_X = exp_X
    n_dims_to_sum = len(X.shape) -1
    for i in range(1, n_dims_to_sum+1): # sum dimensions 1 to n, included
        sum_X = torch.sum(sum_X, dim=i, keepdims=True)

    return exp_X/sum_X



if __name__ == "__main__":
    X = torch.tensor([[0, -np.inf], [-np.inf, 0]], dtype=torch.float32)
    I = torch.tensor(np.eye(2), dtype=torch.float32)
    assert torch.allclose(batch_softmax(X), I)

    X = torch.randn((100,10))
    expected_batch_softmax_X = torch.zeros(X.shape)
    for i in range(X.shape[0]):
        expected_batch_softmax_X[i,:] = torch.softmax(X[i,:], dim=0)
    assert torch.allclose(batch_softmax(X), expected_batch_softmax_X)

    X = torch.randn((100, 10, 10))
    expected_batch_softmax_X = torch.zeros(X.shape)
    for i in range(X.shape[0]):
        exp_X = torch.exp(X[i,:,:])
        expected_batch_softmax_X[i,:,:] = exp_X/(exp_X.sum())
    assert torch.allclose(batch_softmax(X), expected_batch_softmax_X)






#%%
class AttentionCNN(CNN):
    """
    A convolutional network that allows basic operations : early to late fusion,
    using depth-wise, time-wise, or channel-wise concatenation.
    Depending on the input_shape parameter, the convolutions are either 1D or 2D
    """

    def __init__(self, input_shape, n_branches, attention_type):
        """
        Parameters
        ----------
        - input_shape: either a couple (channel, time); or a triple (channel, freq, time)
            the input shape for 1 branch
        - n_branches (int): How many convolution branches to keep. Must be at
            least 1. In case of depthwise concatenation, the different channels
            going to a single branch must be contiguous.
            n_branches = 1 for classic network; >1 for feature fusion
        - attention_type (string): either "selective_fusion" or  "classic"
            (see above for references)

        """

        CNN.__init__(self, input_shape, n_branches, fusion="conv2")
        self.n_signals = n_branches

        if len(input_shape) == 2: # 1D data
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        else: # 2D data
            self.global_pool = nn.AdaptiveAvgPool2d(1)


        self.attention_type = attention_type
        if attention_type == "selective_fusion":
            self.Attention = nn.Linear(self.FC0.in_features, self.FC0.in_features)
        elif attention_type == "classic":
#            self.Attention_k = nn.Linear(self.FC0.in_features, self.FC0.in_features)
#            self.Attention_q = nn.Linear(self.FC0.in_features, self.FC0.in_features)
#            self.Attention_v = nn.Linear(self.FC0.in_features, 1)

            conv = type(self.conv0) # either nn.Conv1d, or nn.Conv2d

#            self.Attention = conv(64, n_branches, 1, groups=n_branches)
            self.Attention = conv(64, 1, 1)
                            # FC along channels == conv with filter size 1
                            # we keep one group per sensor
            global_pool = nn.AdaptiveAvgPool2d if conv == nn.Conv2d else nn.AdaptiveAvgPool1d
            self.global_avg = global_pool(1)
            self.FC0 = nn.Linear(64, 128)
        else :
            raise ValueError(f"Unknown value for attention_type argument: '{attention_type}'.\n Available values: 'selective_fusion', 'classic'")

        self.attention_weights = [] # a list of cuda tensors or numpy arrays




    def forward(self, X, return_weights=False):
        """
        The behaviour of this function depends on the value of the return_weights
        variable.

        Input :
            X : an input tensor. Its shape should be equal to
                (B, input_shape[0] * n_branches, input_shape[1], input_shape[2])
                or
                (B, input_shape[0] * n_branches, input_shape[1])
                depending on input_shape dimension
            return_weights (bool):
                If False, returns a (B, 8) tensor, the prediction for the given batch
                If True, returns the weight values:
                    either a (B, n_signals*64*H*W) tensor ('selective_fusion' attention type)
                    or a     (B, 64, n_signals*H, W) tensor ('classic')
        Output :
            either scores or attention_weights
        """

        X = self.conv0(X)
        X = self.mp(self.relu_conv0(X))
        X = self.conv1(X)
        X = self.mp(self.relu_conv1(X))
        X = self.conv2(X)
        X = self.mp(self.relu_conv2(X))  # shape: (B, n_signals*64, H, W)

        if self.attention_type == "selective_fusion":
            X = X.view(X.shape[0],-1)
            attention_weight = self.Attention(X)    # shape: (B, n_signals*64*H*W)
            attention_weight = torch.sigmoid(attention_weight)  # now between 0 and 1
            X = X*attention_weight

        elif self.attention_type == "classic":
            per_sensor_features = X.chunk(self.n_signals, dim=1) # list of (B, 64, H, W) tensors
            X = torch.cat(per_sensor_features, dim=2)            # shape: (B, 64, n_signals*H, W)
            attention_weight = self.Attention(X)                 # shape: (B, 64, n_signals*H, W)
            attention_weight = batch_softmax(attention_weight)
            X = X*attention_weight

            # now, we sum this into a single (B, 64) tensor.
            # reminder : X can be either 2D or 1D (even though all comments assume X is 2D)
            n_scalars = X.numel()/(X.shape[0]*X.shape[1])      # = 64*n_signals*H*W in the 2D case
                                                               # = 64*n_signals*T   in the 1D case
            X = n_scalars * self.global_avg(X)  # sum = N*mean        #shape: (B, 64, 1, 1)

            X = X.squeeze(2)
            if len(X.shape) == 3: # if X was a 2D signal (instead of a 1D signal)
                X = X.squeeze(2)

        if return_weights: return attention_weight

        if self.training: X = self.dropout0(X)
        X = self.relu_FC0(self.FC0(X))
        if self.training: X = self.dropout1(X)
        X = self.FC1(X)

        return X



    def get_weights(self, signals_list, dataloader):
        """
        Paramters
        ---------
        signals_list: a list of string, used to make the correspondance
            between the weights and the signals
        dataloader: a dataloader to compute the attention weights

        Returns
        -------
        weights: a dictionnary of arrays. Each array is 1-dimensional, with
            64 elements
        """
        coefficients = {}


        attention_weights_list=[]

        self.train(False)

        for X,_ in dataloader:
            attention_weight = self(X, return_weights=True)

            if self.attention_type == "selective_fusion":  #attention_weight.shape = (B, 64*n_signals*H*W)
                attention_weight = attention_weight.view(X.shape[0],64, -1)
            elif self.attention_type == "classic": # attention_weight.shape = (B, 1, n_signals*H, W)
                attention_weight = attention_weight.view(X.shape[0], -1)

            attention_weights_list.append(attention_weight.detach().cpu().numpy())
                # importing each tensor from the GPU is slower than keeping
                # only cuda tensors and importing them at the end, but it
                # requires less GPU memory

        complete_weights = np.concatenate(attention_weights_list, axis=0)        # shape: (batch, channels, T)
        if self.attention_type == "selective_fusion":
            complete_weights = np.mean(np.mean(complete_weights, axis=2), axis=0)  # shape: (channels)
        elif self.attention_type == "classic":
            complete_weights = np.mean(        complete_weights         , axis=0)  # shape: (channels)
#        complete_weights = complete_weights.transpose(1,0,2)                      # shape: (channels, batch, T)
#        complete_weights = complete_weights.reshape(complete_weights.shape[0],-1) # shape: (channels, batch*T)

        n_channels = complete_weights.shape[0] # we removed the batch dimension during the np.mean(), this correspods to the former second dimensoin
        n_signals = len(signals_list)
        per_signal_channel = n_channels//n_signals
            # the division should be exact, we just need an integer
        error_string = f"{n_channels} channels for {n_signals} signals ({signals_list})\nThe division must be exact"

        assert (per_signal_channel*n_signals == n_channels), error_string

        for i, signal in enumerate(signals_list):
            coefficients[signal] = complete_weights[i*per_signal_channel:(i+1)*per_signal_channel].reshape(-1)

        return coefficients





    def plot_weights_boxplot(self, val_dataloader, signals_name_list):
        with torch.no_grad():
            _, val_f1, _, _, _ = self.run_one_epoch(val_dataloader, gradient_descent=False)

        weight_dict = self.get_weights(signals_name_list, val_dataloader)
        weights_array = np.stack(list(weight_dict.values()), axis=-1) # shape: (batch*64, n_signals)

        plt.figure()
        plt.boxplot(weights_array, sym="", labels=signals_name_list)
        plt.title(f"val F1 = {100*val_f1:.3f}% (clean data)")
        plt.show()

        for signal_name in transform.signal_name_list:

            val_dataloader.dataset.data[signal_name + "_copy"] = val_dataset.data[signal_name].clone()
            val_dataloader.dataset.data[signal_name] = val_dataset.data[signal_name]*0. + val_dataset.data[signal_name].mean()

            with torch.no_grad():
                _, val_f1, _, _, _ = self.run_one_epoch(val_dataloader, gradient_descent=False)

            weight_dict = self.get_weights(transform.signal_name_list, val_dataloader)
            weights_array = np.stack(list(weight_dict.values()), axis=-1) # shape: (batch*64, n_signals)

            plt.figure()
            plt.boxplot(weights_array, sym="", labels=transform.signal_name_list)
            #plt.violinplot(weights_array) #, labels=transform.signal_name_list)
            plt.title(f"val F1 = {100*val_f1:.3f}% ({signal_name} set to mean)")
            plt.show()


            val_dataloader.dataset.data[signal_name] = val_dataset.data[signal_name + "_copy"].clone()
            del val_dataloader.dataset.data[signal_name + "_copy"]


        if self.attention_type == "selective_fusion":
            plt.figure()
            att = self.Attention.weight.data.cpu().numpy()
            max_value = np.abs(att).max()
            factor = 10
            plt.imshow(factor*att, vmin=-max_value, vmax=max_value)
            plt.title(f"weights of the attention matrix ({factor}x)")
            plt.colorbar()

            n_signals = len(signals_name_list)
            signal_range = att.shape[0]//n_signals
            pos = np.linspace(0, signal_range*(n_signals-1), n_signals) + signal_range//2
            plt.yticks(pos, labels=signals_name_list)
            plt.xticks(pos, labels=signals_name_list)
            plt.ylabel("input")
            plt.xlabel("output")

        plt.show()


    def plot_example(self, X, Y, list_signal_names):

        n_signals = X.shape[1]
        with torch.no_grad():
            attention_weight = self(X, return_weights=True)
            _, predicted_class = self.predict(X)

        predicted_class_name = classes_names[predicted_class.item()]
        GT_class_name = classes_names[Y.item()]
        title_str = f"GT: {GT_class_name}\npredicted: {predicted_class_name}\n"

        X = X.detach().cpu().numpy()                # shape: (1, n_signals, 48, 48)
        attention_weight = attention_weight.detach().cpu().numpy()   # shape: (1, 1, n_signals*H, W)
        per_signal_attn_weight = np.split(attention_weight,n_signals, axis=2) # list of (1, 1, H, W) arrays

        plt.figure()
        vmin_attn = min([attn_weight.min() for attn_weight in per_signal_attn_weight])
        vmax_attn = max([attn_weight.max() for attn_weight in per_signal_attn_weight])

        for i_signal in range(n_signals):
            plt.subplot(2, n_signals, i_signal+1)
            plt.imshow(X[0, i_signal, :,:], vmin=X.min(), vmax=X.max())
            plt.title(title_str+list_signal_names[i_signal])
            plt.colorbar()

            plt.subplot(2 ,n_signals, i_signal+n_signals+1)
            plt.imshow(per_signal_attn_weight[i_signal][0,0,:,:], cmap="gray", vmin=vmin_attn, vmax=vmax_attn)
            plt.title("attention \n" +list_signal_names[i_signal])
            plt.colorbar()

        plt.show()


model = AttentionCNN(input_shape=(1, 48, 48), n_branches=2, attention_type='selective_fusion')


#%%
if __name__ == "__main__":

    import torch.utils.data
    from preprocess import Datasets
    from preprocess.transforms import SpectrogramTransform
    from preprocess.fusion import ConcatCollate
    from architectures.base_network import device


    from param import fs, duration_window, duration_overlap, spectro_batch_size


    # unit test: Hide a quarter of the spectrogram, and see the attention decrease there
    transform = SpectrogramTransform(["Acc_norm", "Gyr_y"], fs, duration_window, duration_overlap, spectro_batch_size,
                                                 interpolation='log', log_power=True, out_size=(48,48))

    signals_name_list = transform.signal_name_list
    collate_fn = ConcatCollate("depth", signals_name_list)   # concatenate signals and merge into a Batch

    def black_square_collate(L):
        X, Y = collate_fn(L)

        mask = torch.zeros_like(X) + 1.
        mask[:,:,24:,24:] = 0

        return X*mask, Y

    try :  # do not reload the datasets if they already exist
        train_dataset
        val_dataset

    except NameError:
        train_dataset = Datasets.SignalsDataSet(mode='train', split='balanced', comp_preprocess_first=True, transform=transform)
        val_dataset =   Datasets.SignalsDataSet(mode='val',   split='balanced', comp_preprocess_first=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, collate_fn=black_square_collate, shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, collate_fn=black_square_collate, shuffle=False)

    model = AttentionCNN(input_shape=(1, 48, 48), n_branches=2, attention_type='classic')
    model.to(device)

#        model.plot_per_class_weight(transform.signal_name_list, val_dataloader)

    import matplotlib.pyplot as plt

    _, _, _, val_f1 = model.train_process(train_dataloader, val_dataloader, maxepochs=50)

    #%%




    if model.attention_type == "selective_fusion":
        model.plot_weights_boxplot(val_dataloader, signals_name_list)

    elif model.attention_type == "classic":
        model.plot_weights_boxplot(val_dataloader, signals_name_list)


        for _ in range(5):
            i = np.random.randint(0,len(val_dataloader.dataset))
            example_sample = val_dataset[i]
#            model.plot_example(example_sample, signals_name_list)

            X, Y = black_square_collate([example_sample])
            model.plot_example(X, Y, signals_name_list)

            X, Y = collate_fn([example_sample])
            model.plot_example(X, Y, signals_name_list)

