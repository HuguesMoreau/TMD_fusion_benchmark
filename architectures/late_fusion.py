"""
Author Hugues

This file contains late fusion (average of scores or probas)
"""


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


from param import device, classes_names
from architectures.base_network import Network
from architectures.basic_CNN import CNN
from visualizations.classes import colors as color_per_class




def softmax_sum(tensors_dict, weights_dict):
    """
    Compute a weighted average of the tensors in tensors_dict, here the weights
    are given by a softmax of the scalars given in weights_dict

    Parameters
    ----------
    tensors_dict: a dict of tensors
        can be either scores or probas (shape: (batch_size, n_classes))
    weights_dict: a dict of nn.Linear layers with weights shape: (1,1)
        Both dictionaries are expected to have the same keys

    Returns
    -------
    a single tensor with the same shape as the input tensors
        ((batch_size, n_classes) by default)
    """
    example_tensor = list(tensors_dict.values())[0]
    example_shape = example_tensor.shape

    max_weight = -np.inf
        # we can add any value to the inputs of a softmax without changing the
        # outputs, so we substract the highest value, to avoit numerical errors
    for signal in tensors_dict.keys():
        this_weight_cpu = weights_dict[signal].weight.item()
        if this_weight_cpu > max_weight:
            max_weight = this_weight_cpu
    max_weight = torch.tensor([max_weight], device=device, dtype=torch.float)

    sum_tensors = torch.zeros(example_shape, device=device)
    sum_weights = torch.zeros(1, device=device)
    for signal in tensors_dict.keys():
        this_weight = weights_dict[signal].weight
        sum_tensors = sum_tensors + torch.exp(this_weight-max_weight) * tensors_dict[signal]
        sum_weights = sum_weights + torch.exp(this_weight-max_weight)
    return sum_tensors/sum_weights




if __name__ == "__main__":
    # we use softmax_sum to compute a weighted sum of one-hot vectors.
    # As the i-th vector is equal to zero everywhere, except that its i-th
    #  coordinate is a one, the value of the softmax_sum of tensors is a single
    # tensor, which values are equal to the softax of weights

    model_names = ["A", "B", "C", "D"]
    weights_list = [(2*torch.randn(1, device=device)) for m_name in model_names]
    weights_dict = {model_names[i]:weights_list[i] for i in range(len(weights_list))}

    weight_layers = {m_name: nn.Linear(1,1).to(device) for m_name in model_names}
    for model_name in model_names:
        weight_layers[model_name].weight.data = weights_dict[model_name]

    X_one_hot = {m_name:torch.zeros(len(weights_list), device=device) for m_name in model_names}
    for i in range(len(weights_list)):
        X_one_hot[model_names[i]][i] = 1

    softmax_fn = softmax_sum(X_one_hot, weight_layers)

    weights_tensor = torch.tensor(weights_list, device=device)
    softmax_torch = torch.softmax(weights_tensor, dim=0)

    assert torch.allclose(softmax_fn, softmax_torch)

#%%

def plot_bar_per_class(X, Y, color_per_class, n_bins):
    """
    Draws a stacked histogram where the colors are the classes.
    Will be used later, in plot_predictions_histograms

    Parameters
    ----------
    X (1D np array): the values to plot in the histogram
    Y (1D np array of ints): the classes of each sample
    color_per_class: a list of colors for the legend (one color = 3-list)
    n_bins: (int) number of bins in the hsistogram

    Returns None
    """
    bins=np.linspace(0.,1., n_bins)
    bin_centers = (bins[1:] + bins[:-1])/2
    bottom = np.zeros(bin_centers.shape)

    for i in range(Y.max()):
        this_class_X = X[Y==i]
        histogram, _ = np.histogram(this_class_X, bins=bins)

        plt.bar(x=bin_centers, height=histogram, bottom=bottom,
                 width=1/n_bins, color=color_per_class[i])
        bottom += histogram







    #%%

class LateFusionCNN(Network):
    """
    A convolutional network that allows to use (weighted) sums of probabilities
    or scores
    """

    def __init__(self, input_shape, signals_list, fusion_mode, use_weights):
        """
        Inputs :
            - input_shape (triple of ints, (channel, freq, time))
                If there is only one input channel, the next two arguments
                (n_signals and fusion) are ignored
            - signals_list (list of strings): Each string is the name of one
                network. The names need to correspond to the names given to the
                transform funtion (ex: 'Acc_norm', 'Gyr_y')
                To use the same signal several times, add '_copy', followed by
                a unique index after the signal name (eg 'Acc_norm_copy_10',
                'Acc_norm_copy_1337', 'Acc_norm_copy_42')
            - fusion_mode (string, one of 'scores', 'probas')
            - use_weights (bool): if True, replaces the mean with a weighted
                mean, where the weights are updated by gradient descent
           """
        super(LateFusionCNN, self).__init__()

        self.signals_list = signals_list
        self.networks = {}  # a dictionnary containing CNN as values
        self.weight_layers = {}  # if we dont use weights, this dictionnary will
                                 # hold constant values

        for signal in signals_list:
            specific_CNN = CNN(input_shape, n_branches=1, fusion="start")
            self.networks[signal] = specific_CNN
            net_name = signal + "_CNN"
            self.add_module(net_name, specific_CNN)

            weight_layer = nn.Linear(1,1, bias=False)
            weight_layer.weight.data = torch.FloatTensor([1.])  # Although the
                # layer is called 'Linear', we will use a softmax of these weights.
                # See the softmax_sum function

            self.weight_layers[signal] = weight_layer
            if use_weights: self.add_module(signal + "_weight", weight_layer)
            # update the weights by gradient descent

        self.use_weights = use_weights
        self.fusion_mode = fusion_mode

        if fusion_mode == "probas":
            # the CrossEntropy score works with scores, but we will need a loss
            # that uses probabilities: the NLLLoss.
            # reminder:      CrossEntropyLoss(X, Y) = NLLLoss(softmax(X, Y))
            # We replace the default loss (CrossEntropyLoss) by the NLLLoss
            weight_loss = self.criterion.weight
            self.criterion = nn.NLLLoss(weight = weight_loss)



        self.threshold = np.log(float(2**60))  #  maximum value a float can take before its exponential overflows
        # A float (simple precision) Has 8 bits for the exponent, but one of those
        # bits is dedicated to the sign
        # In theory, a float overflows at 2**(2**7 -1 -1)  = 2**126
        # (the second -1 is here because an exponent of only ones is dedicated
        # to special values), but Pytorch cannot go further than
        # 2**(2**6 -2) = 2**62 without issuing an error (overflow)
        # This will lead small numerical errors (probabinities cannot be lower
        # than 0.03, for instance)

    def to(self, device):
        """
        We overridte the torch.nn.Module.to method so that the weights are sent
        to the GPU even if they are not in self.children() (which happens
        when self.uwe_weights = False)

        This method has the same signature as the overriden  method
        """
        for weight_layer in self.weight_layers.values():
            weight_layer = weight_layer.to(device)
        for p in self.children():
            p.to(device)

        return self


    def forward(self, X):
        """
        Parameters
        ----------
        X : a dictionnary of (B, F, T) input tensors. The keys of this
            dictionnary need to be the same as the keys of self.networks

        Returns
        -------
        a (B, 8) tensor
            /!\ If self.fusion_mode == "probas", the function returns a log of
            probability, so that the NLLoss can apply to the output. (this is
            different from all other forward methods, which return scores). As
            the log and softmax are increasing functions, this does not affect
            the behaviour of predict().
            If self.fusion_mode == "scores", the function returns a score,
            as usual
        """

        scores_dict = {}
        for signal in self.signals_list:
            if '_copy' in signal: # cut everything after the '_copy'
                _copy_index = signal.find('_copy')
                root_signal = signal[:_copy_index]
            else:
                root_signal = signal

            network = self.networks[signal]
            scores_dict[signal] = network(X[root_signal])

        if self.fusion_mode == 'scores':
            scores = softmax_sum(scores_dict, self.weight_layers)
            return scores

        else :  #self.fusion_mode == 'probas'

           # If we computed the probabilities right now, we would have numerical
           # stability problems, for some probabilities will be exactly one, which
           # couse the loss to be exactly zero, which makes the gradient undefined.
           # We use two things to correct this:

           # 1) Rescale the scores so that the new maximum and the new minimum
           # are under zero (we do not really cre if the minimum 'underflows')
           # 2) use a sigmoid to prevent the scores from overflowing, except
           # that the output values are between -2**64 and 2**64.
           # the sigmoid is the continuous equivalent of np.clip
           probas_dict = {}
           for signal in self.signals_list:
               # maximum value a float can take before its exponential overflows: 2**(2**8) = 2**64

               rescaled_score = torch.sigmoid((scores_dict[signal]-torch.max(scores_dict[signal], dim=1, keepdims=True)[0])/self.threshold)
                           # between 0 and 0.5
               rescaled_score =  (rescaled_score-0.5) *self.threshold
                           # between -threshold and 0


#               rescaled_score = scores_dict[signal]
               probas_dict[signal] = torch.softmax(rescaled_score, dim=1)

           probas = softmax_sum(probas_dict, self.weight_layers)
           return torch.log(probas)

#            probas_dict = {signal:torch.softmax(scores_dict[signal], dim=1) for signal in self.signals_list}




    def get_weights(self, signals_list=None, dataloader=None):
        """
         we use softmax_sum to compute a weighted sum of one-hot vectors.
         As the i-th vector is equal to zero everywhere, except that its i-th
          coordinate is a one, the value of the softmax_sum of tensors is a single
         tensor, which values are equal to the softax of weights

        Paramters
        ---------
        signals_list (optional): a list of string, used to order the weights.
            If None, self.signals_list is used instead
            default: None
        dataloader(optional): an argument to mimick the signature of
            AttentionCNN.get_weights.
            It is systematically ignored

        Returns
        -------
        coefficients: a dictionnary of floats
        """
        if signals_list==None: signals_list = self.signals_list

#        ones = {s_name:torch.ones(len(signals_list), device=device) for s_name in signals_list}
        X_one_hot = {signal:torch.zeros(len(signals_list), device=device) for signal in signals_list}
        for i in range(len(signals_list)):
            X_one_hot[signals_list[i]][i] = 1

        coefficients_tensor = softmax_sum(X_one_hot, self.weight_layers)
        coefficients_dict = {}
        for i, signal_name in enumerate(signals_list):
            coefficients_dict[signal_name] = coefficients_tensor[i].item()
        return coefficients_dict



    def train_process(self, train_dataloader, val_dataloader, maxepochs, proportion_separate=0.5):
        """
        Overrides the base training process to train the subnetworks half the
        time, and the complete network the other half of the time

        Parameters
        ----------
        - train_dataloader (pytorch DataLoader object)
        - val_dataloader (pytorch DataLoader object)
        - maxepochs (int)
        - proportion_separate (float, between 0 and 1): the proportion of epochs
            during which the individual modela are trained separately, before
            training the whole model

        Returns
        -------
        (the same outputs as the base train_process)

        """
        maxepochs_separate = int(round(proportion_separate * maxepochs ))
        maxepochs_ensemble = maxepochs - maxepochs_separate

        if maxepochs_separate > 0 :
            global_collate_fn = train_dataloader.collate_fn
            for signal in self.signals_list:
                print(signal)

                if '_copy' in signal: # cut everything after the '_copy'
                    _copy_index = signal.find('_copy')
                    root_signal = signal[:_copy_index]
                else:
                    root_signal = signal

                # we temporarily replace the collate function of the dataloader
                def select_signal(L):
                    (X, Y) = global_collate_fn(L)
                    return X[root_signal], Y
                train_dataloader.collate_fn = select_signal
                val_dataloader.collate_fn   = select_signal


                Network.train_process(self.networks[signal], train_dataloader, val_dataloader, maxepochs_separate)

            train_dataloader.collate_fn = global_collate_fn
            val_dataloader.collate_fn   = global_collate_fn

        print('Global model')
#        for i in range(maxepochs_ensemble):
#            results = Network.train_process(self, train_dataloader, val_dataloader, 1)
#
#            input_tensors = {signal_name:torch.tensor([[1.]], dtype=float, device=device) for signal_name in self.signals_list}
#            coefficients = softmax_sum(input_tensors, self.weight_layers)
#            print(coefficients.to(torch.device('cpu')).detach().numpy())

        if maxepochs_ensemble >0 :
            results = Network.train_process(self, train_dataloader, val_dataloader, maxepochs_ensemble)
        else:
            results = Network.train_process(self, [],               val_dataloader, 1)
            #the training loss and f1 will be equal to -1

        # print the final coefficients
        coefficients = [self.weight_layers[signal].weight.data for signal in self.signals_list]
        coefficients = torch.cat(coefficients)
        print(self.signals_list)
        print(torch.softmax(coefficients, dim=0).to(torch.device('cpu')).detach().numpy())

        return results




    def test(self, train_dataloader, val_dataloader, test_dataloader, maxepochs, proportion_separate=0.5):
        """
        trains the model on {train + val}, and tests it on {test} once

        Parameters
        ----------
        train_dataloader (pytorch DataLoader object)
        val_dataloader (pytorch DataLoader object)
        test_dataloader (pytorch DataLoader object)
        proportion_separate (float, between 0 and 1): the proportion of epochs
            during which the individual modela are trained separately, before
            training the whole model


        Returns
        -------
        a 6-tuple of:
        train_loss (positive float)
        val_loss   (positive float)
        test_loss  (positive float)
        train_f1 (float, between 0. and 1.)
        val_f1   (float, between 0. and 1.)
        test_f1  (float, between 0. and 1.)

        The train and val scores are from the last epoch
        """
        print("\nTraining the model:")
        print(self)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)

        maxepochs_separate = int(round(proportion_separate * maxepochs ))
        maxepochs_ensemble = maxepochs - maxepochs_separate

        if maxepochs_separate > 0 :
            global_collate_fn = train_dataloader.collate_fn
            for signal in self.signals_list:
                print(signal)

                if '_copy' in signal: # cut everything after the '_copy'
                    _copy_index = signal.find('_copy')
                    root_signal = signal[:_copy_index]
                else:
                    root_signal = signal
                print('after',train_dataloader.collate_fn)

                # we temporarily replace the collate function of the dataloader
                def select_signal(L):
                    (X, Y) = global_collate_fn(L)
                    return X[root_signal], Y
                train_dataloader.collate_fn = select_signal
                val_dataloader.collate_fn   = select_signal
                test_dataloader.collate_fn  = select_signal

                Network.test(self.networks[signal], train_dataloader, val_dataloader, test_dataloader, maxepochs_separate)

            train_dataloader.collate_fn = global_collate_fn
            val_dataloader.collate_fn   = global_collate_fn
            test_dataloader.collate_fn  = global_collate_fn

        print('Global model')
        if maxepochs_ensemble >0 :
            results = Network.test(self, train_dataloader, val_dataloader, test_dataloader, maxepochs_ensemble)
        else:
            results = Network.test(self, [],               [],             test_dataloader, 1)
            #the training loss and f1 will be equal to -1

        return results








#%%
    def collect_probabilities(self, dataloader):
        """
        Records the output probabilities assigned to the input samples by each
        individual model plus the global model

        Parameters
        ----------
        dataloader: a pytorch DataLoader object containing the input samples to evaluate

        Returns
        -------
        probabilities: dict
            keys: strings (either a signal name or "Global\model")
            values: Couple of
                2-dimensional np array of floats between 0 and 1 (the probabilities themselves, shape: (batch, n_classes))
                1-dimensional np array of ints between 0 and n_classes-1 (the classes)
                    We return the classes in case the dataloader shuffles the data
        """
        probabilities = {}

        global_collate_fn = dataloader.collate_fn
        for signal in self.signals_list:
            if '_copy' in signal: # cut everything after the '_copy'
                _copy_index = signal.find('_copy')
                root_signal = signal[:_copy_index]
            else:
                root_signal = signal

            # we temporarily replace the collate function of the dataloader
            def select_signal(L):
                (X, Y) = global_collate_fn(L)
                return X[root_signal], Y
            dataloader.collate_fn = select_signal
            with torch.no_grad():
                _, _, _, scores, Y = self.networks[signal].run_one_epoch(dataloader, gradient_descent=False)

            # cap scores to prevent overflow (simple precision)
            scores = np.clip(scores, -30, 30)
            this_model_probas = np.exp(scores) / (np.sum(np.exp(scores), axis=1, keepdims=True) )
            probabilities[signal] = (this_model_probas, Y)

        # Global model
        dataloader.collate_fn = global_collate_fn
        if self.fusion_mode == "probas":
            with torch.no_grad():
                _, _, _, log_softmax, Y = self.run_one_epoch(dataloader, gradient_descent=False)
            global_probas = np.exp(log_softmax) #/ (np.sum(np.exp(log_softmax), axis=1, keepdims=True))

        elif self.fusion_mode == "scores":
            with torch.no_grad():
                _, _, _, scores, _ = self.run_one_epoch(dataloader, gradient_descent=False)
            scores = np.clip(scores, -30, 30)
            global_probas = np.exp(scores) / (np.sum(np.exp(scores), axis=1, keepdims=True) )
        probabilities["Global\nmodel"] = (global_probas, Y)

        return probabilities



#%%
    def plot_predictions_histograms(self, dataloader):
        """
        Displays the probabilities of each sensor-specific model in a histogram

        Parameters
        ----------
        dataloader: a pytorch DataLoader object containing the input samples to evaluate

        Returns None
        """
        probabilities = self.collect_probabilities(dataloader)
        n_bins = 20

        example_score = list(probabilities.values())[0][0]
        n_classes = example_score.shape[1]  # we assume all models produce the same number of classes
        n_signals = len(self.signals_list) +1 #we include the global model n the signals
        n_models = len(probabilities)
        plt.figure()
        for i_signal, signal in enumerate(self.signals_list+["Global\nmodel"]):
            probas, Y = probabilities[signal]
            for i_class in range(n_classes):
                plt.subplot(n_classes, n_models+1, i_class*(n_models+1) +i_signal  +1 )
                plot_bar_per_class(probas[:,i_class], Y, color_per_class, n_bins)
                plt.yticks([],[])
                plt.xticks(ticks=np.linspace(0,1,6), labels=['']*6, fontsize=6)
                if i_class  == 0: plt.title(signal, fontsize=8)
                if i_signal == 0: plt.ylabel(classes_names[i_class], fontsize=8)
                if i_class == n_classes-1: plt.xticks(ticks=np.linspace(0,1,6),
                                                      labels=[f'{x:.1f}' for x in np.linspace(0,1,6)])

        # add the legend in the last column
        plt.subplot(n_classes, n_models+1, 0*(n_models+1) +n_signals +1 )
        labels_legend = ["Actual class"] + classes_names # add one label for legend title
        colors_legend = [[1,1,1,0]] + [color+[1] for color in color_per_class]# add an alphachannel for transparency
        bars_legend = [matplotlib.patches.Patch(facecolor=color, edgecolor=color) for color in colors_legend]
        plt.legend(bars_legend, labels_legend, fontsize=6)
        plt.xticks([],[]); plt.yticks([],[]) # erase ticks
        plt.axis('off') # erase axis



    def plot_confusion_between_models(self, dataloader):
        """
        Displays the probabilities of each sensor-specific model in a histogram
        /!\ This function assumes the dataloader has shuffle = False

        Parameters
        ----------
        dataloader: a pytorch DataLoader object containing the input samples to evaluate

        Returns None
        """
        probabilities = self.collect_probabilities(dataloader)

        predictions = {signal:np.argmax(results[0], axis=1) for (signal, results) in probabilities.items()}

        plt.figure()
        all_models = list(predictions.keys())
        n_models = len(predictions)
        cross_acc_scores = np.zeros((n_models, n_models))
        for i1, signal_1 in enumerate(predictions.keys()):
            for i2, signal_2 in enumerate(predictions.keys()):
                accuracy = np.mean(predictions[signal_1] == predictions[signal_2])
                cross_acc_scores[i1, i2] = accuracy


        for i in range(n_models):
            for j in range(n_models):
                text = np.round(cross_acc_scores[i, j], 3)
                colour = "w" if cross_acc_scores[i, j]<0.5 else "k"
                plt.text(j, i, text, ha="center", va="center", color=colour, size=10)
        plt.imshow(cross_acc_scores, vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks(np.arange(len(all_models)),all_models)
        plt.yticks(np.arange(len(all_models)),all_models)
        plt.title("cross_model_accuracies")

    #%%
if __name__ == "__main__":
    import torch.utils.data
    from preprocess import Datasets
    from preprocess.transforms import SpectrogramTransform
    from preprocess.fusion import separate_sensors_collate
    from architectures.base_network import device
    from param import fs, duration_window, duration_overlap, spectro_batch_size

    spectrogram_transform = SpectrogramTransform(["Acc_norm", "Gyr_y", "Mag_norm", "Ori_w"], fs, duration_window, duration_overlap,
                                                 spectro_batch_size, interpolation='log', log_power=True, out_size=(48,48))
    collate_fn = separate_sensors_collate

    try :  # do not reload the datasets if they already exist
        train_dataset
        val_dataset

    except NameError:
        train_dataset = Datasets.SignalsDataSet(mode='train', split='balanced', comp_preprocess_first=True, transform=spectrogram_transform)
        val_dataset =   Datasets.SignalsDataSet(mode='val',   split='balanced', comp_preprocess_first=True, transform=spectrogram_transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, num_workers=0, shuffle=True)
    train_dataloader_noshuffle = torch.utils.data.DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, num_workers=0, shuffle=False)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, collate_fn=collate_fn, num_workers=0)

    model = LateFusionCNN(input_shape=(1,48,48), signals_list=["Acc_norm", "Gyr_y", "Mag_norm", "Ori_w"],
                        fusion_mode="probas", use_weights=False)


    model.to(device)


    # Investigate why the probabilities fusion method overfits:
    model.train_process(train_dataloader, val_dataloader, maxepochs=25, proportion_separate=1)
    model.plot_predictions_histograms(val_dataloader)
    model.plot_confusion_between_models(train_dataloader_noshuffle)
    model.plot_confusion_between_models(val_dataloader)
    model.train_process(train_dataloader, val_dataloader, maxepochs=25, proportion_separate=0.)
    model.plot_predictions_histograms(val_dataloader)
    model.plot_confusion_between_models(train_dataloader_noshuffle)
    model.plot_confusion_between_models(val_dataloader)
    print(model.get_weights())


