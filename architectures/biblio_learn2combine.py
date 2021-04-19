"""
Author Hugues

A third-party (unofficial) implementation of :
Liu, Kuan, Yanen Li, Ning Xu, et Prem Natarajan.
« Learn to Combine Modalities in Multimodal Deep Learning ».
arXiv:1805.11730 [cs, stat], May 29, 2018. http://arxiv.org/abs/1805.11730.

"""


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn

from architectures.base_network import Network
from architectures.late_fusion import LateFusionCNN
from param import device


#%%


class L2C_Loss(nn.Module):
    def __init__(self, weights, beta, delta):
        """
        Creates a loss to take into account the q_i, and the boosting extension

        Parameters
        ----------
        weights (torch.tensor): the per-class weights. The weights will be
            normalizedwhen the loss function is called
        beta (float): the intensity of the down-weighing.
            The paper advises to keep this parameter between 0 and 1 (the
            optimum seems to be 0.5), but it coult take any value
        delta (float): the threshold of the boosting extension. if delta =
            np.inf, the boosting extension has no effect.

        Attributes
        ----------
        base_loss
        beta: a (1,) tensor containing the input value
        delta: a (1,) tensor containing the input value
        """
        super(L2C_Loss, self).__init__()
        self.per_class_weight = weights/weights.sum()
        self.beta = torch.tensor(beta, dtype=torch.float, device=device)
        self.delta = torch.tensor(delta, dtype=torch.float, device=device)


    def forward(self, probas, target, reduce="mean"):
        """
        Parameters
        ----------
        probas (torch.FloatTensor): probability tensor, with shape
            (batch_size, n_classes, n_models)
        target (torch.LongTensor): the ground truth, integers between 0 and n-1
            shape: (batch_size,)
        reduce (string, optional): either 'none' or 'mean'
            Similar to the pytorch 'reduction' argument, except that it appears
            in the forward method instead of __init__. The reason being that we
            need to compute one loss per sample in the L2C_CNN.predict method,
            so that we can attribute to each sample the class that minimize the
            loss.
            Defaults to "mean".

        Returns
        -------
        loss (torch.tensor): a (1,) tensor to call backwards() on

        """

#        self.probas = probas
#        self.target = target
        batch_size = probas.shape[0]

        M = probas.shape[2]  # number of models
        one = torch.ones(probas.shape, device=device, dtype=torch.float)
        epsilon = 1e-15
        q = torch.prod(one - probas, dim=2, keepdims=True) / (one - probas + epsilon)
        w = torch.pow(q, self.beta/(M-1))

        individual_losses = -torch.log(probas) * w               #shape: (batch_size, n_classes, n_models)
        aggregated_losses = torch.sum(individual_losses, dim=2)  #shape: (batch_size, n_classes)

        losses_GT = aggregated_losses[range(batch_size), target]  #shape: (batch_size,)

        # boosting extension: check if the loss for GT is above all others
        is_above = losses_GT.unsqueeze(1) > aggregated_losses + self.delta

        is_above[range(batch_size), target] = 1    # this will make the computation of the condition easier
        learning_stops = torch.prod(is_above, dim=1)  #shape : (batch_size,)
            #  1 if the loss of the correct class is above all others, 0 otherwise

        mask = 1-learning_stops
        this_weights = mask * self.per_class_weight[target]

        if reduce == 'mean':
            average_loss = torch.sum(this_weights * losses_GT)/torch.sum(this_weights)

        elif reduce == 'none':
            average_loss = this_weights * losses_GT
        else:
            raise ValueError(f"Unknown reduce argument '{reduce}'")

        return average_loss




if __name__ == "__main__":

    """
    The custom loss should be equal to the NLLLoss when beta = 0 and
    delta =+np.inf
    """

    n_classes = 10
    batch_size = 67
    n_models = 4

    weights = torch.softmax(torch.randn((n_classes,), device=device), dim=0)
    target = torch.randint(0, n_classes-1, (batch_size,), device=device)
    scores = torch.randn((batch_size, n_classes, n_models), device=device)
    probas = torch.softmax(scores, dim=1)

    l2c_loss = L2C_Loss(weights, beta=0, delta = np.inf)


    for reduction_method in ['none', 'mean']:
        nll_loss = nn.NLLLoss(weights, reduction=reduction_method)
        product_probas = probas.prod(dim=2)
            #a sum of losses is equivalent to a product of probabilities

        torch_loss_value = nll_loss(torch.log(product_probas), target)

        custom_loss_value = l2c_loss(probas, target, reduce=reduction_method)

        assert (torch.allclose(torch_loss_value, custom_loss_value))




#%%

class L2C_CNN(LateFusionCNN):
    def __init__(self, input_shape, signals_list, beta, delta):
        """
        Inputs :
        input_shape (triple of ints, (channel, freq, time))
            If there is only one input channel, the next two arguments
            (n_signals and fusion) are ignored
        signals_list (list of strings): Each string is the name of one
            network. The names need to correspond to the names given to the
            transform funtion (ex: 'Acc_norm', 'Gyr_y')
        beta (float): the intensity of the down-weighing.
            The paper advises to keep this parameter between 0 and 1 (the
            optimum seems to be 0.5), but it coult take any value
        delta (float): the threshold of the boosting extension. if delta =
            np.inf, the boosting extension has no effect.

           """
        LateFusionCNN.__init__(self, input_shape, signals_list, fusion_mode='probas', use_weights=False)
        # the fusion_mode and use_weights arguments do not actually matter

        print(f"Learn to combine modalities in mutltmodal deep learning, 𝛿={delta}, 𝛽={beta}")

        loss_weights = self.criterion.weight
        self.criterion = L2C_Loss(loss_weights, beta, delta)

        self.threshold = self.threshold = np.log(float(2**20))

    def train_process(self, train_dataloader, val_dataloader, maxepochs=50):
        # we define back the train process we overrode
        return Network.train_process(self, train_dataloader, val_dataloader, maxepochs)





    def forward(self, X):
        """
        Parameters
        ----------
        X : a dictionnary of (B, F, T) input tensors. The keys of this
            dictionnary need to be the same as the keys of self.networks

        Returns
        -------
        probas_array (B, 8, M) proabilities tensor, where M is the number of
            models (also the number of keys in X), and, for each m in range(M),
            probas_array[:,:,m] is the (B,8) probability array of the model
        """

        probas_dict = {}
        for signal in self.signals_list:
            if '_copy' in signal: # cut everything after the '_copy'
                _copy_index = signal.find('_copy')
                root_signal = signal[:_copy_index]
            else:
                root_signal = signal

            network = self.networks[signal]
            score = network(X[root_signal])

            score = score-torch.max(score, dim=1, keepdims=True)[0] # softmax is invariant to the addition of a constant
            rescaled_score = (torch.sigmoid(score/self.threshold)-0.5)*2 * self.threshold

            probas_dict[signal] = torch.softmax(rescaled_score, dim=1)

        probas_array = torch.stack(list(probas_dict.values()), dim=2)

        return probas_array





    def predict(self, X):
        """
        We compute one loss per class, and choose the class with minimal loss

        Parameter (same as forward)
        ---------
        X : a dictionnary of (B, F, T) input tensors. The keys of this
            dictionnary need to be the same as the keys of self.networks


        Returns
        -------
        probas_array: (B, 8, M) proabilities tensor, where M is the number of
            models (also the number of keys in X), and, for each m in range(M),
            probas_array[:,:,m] is the (B,8) probability array of the model
        classes: (batch_size, )  torch LongTensor
        """
        probas = self(X)
        batch_size, n_classes, _ = probas.shape

        losses_tensor = torch.zeros((batch_size, n_classes), device=device,
                                    dtype=torch.float, requires_grad=False)

        for i_class in range(n_classes):
            class_i = i_class * torch.ones((batch_size,), device=device,
                                    dtype=torch.long, requires_grad=False)

            loss_class_i = self.criterion(probas, class_i, reduce='none')
            loss_class_i = loss_class_i.detach()

            losses_tensor[:,i_class] = loss_class_i
            self.optim.zero_grad()

        classes = torch.argmax(-losses_tensor, dim=1)  # min(X) = max(-X)

        return probas, classes







    #%%
if __name__ == "__main__":
    from preprocess import Datasets
    import torch.utils.data
    from preprocess.transforms import SpectrogramTransform
    from preprocess.fusion import separate_sensors_collate
    from param import fs, duration_window, duration_overlap, spectro_batch_size

    # test
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
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, collate_fn=collate_fn, num_workers=0)


    model = L2C_CNN(input_shape=(1,48,48), signals_list=["Acc_norm", "Gyr_y", "Mag_norm", "Ori_w"], beta=0.5, delta=np.inf)

    model.to(device)
    model.train_process(train_dataloader, val_dataloader, maxepochs=50)

