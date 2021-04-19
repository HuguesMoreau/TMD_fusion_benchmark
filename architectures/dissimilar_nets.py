"""
Author Hugues

This file contains the implementation for decorrelated networks (original creation).
The idea was to force the sensor-specific models to learn features that are complementary,
with the idea that similar features will produce similar results.


Alas, the implementation did not work: when the influence of the decorrelation
loss is small, the performance of the decorrelated net is the same (statistically)
as a mere feature concatenation. When the loss becomes higher than a certain
threshold (1.0 for classical CCA, 10 for deep CCA), the performance tanks.
Surprisingly, the same is true when we force the network to produce correlated
features (the loss coefficient can be negative).

We think this is ude to the fact the hypothesis is not realized in practice.
If it were true, we would see that the decorrelation loss is opposed to the
classification loss: the cosine similarity of their gradients would be negative.

Using the plot_conflict argument, one can see the cosine similarity between the
decocrrelation loss and the cross-entropy loss is either mostly positive (with
classical CCA) or symetric around zero (with deep CCA).

Thus, we think that a network can choose itself the adequate level of correlation
between its features, and interfering with it (having a decorrelation loss
that is non-negligible numerically, whether positive or negative) only hurts the
performance
"""


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.metrics import f1_score

from param import device, classes_names
from visualizations.classes import colors as color_classes

from architectures.base_network import Network
from architectures.basic_CNN import CNN



from sklearn.decomposition import PCA
from math import ceil

from external_repositories.svcca import cca_core
from external_repositories.deepCCA.objectives import cca_loss



  #%%


class Correlator(nn.Module):
    """ https://github.com/Michaelvll/DeepCCA """
    def __init__(self, input_features):
        super(Correlator, self).__init__()
        self.out_features = 128
        self.model1 = nn.Sequential(nn.Linear(input_features, self.out_features),
                                    nn.BatchNorm1d(num_features=self.out_features, affine=False),
                                    )
        self.model2 = nn.Sequential(nn.Linear(input_features, self.out_features),
                                    nn.BatchNorm1d(num_features=self.out_features, affine=False),
                                    )

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-3)
        self.criterion = cca_loss(outdim_size=self.out_features, use_all_singular_values=False, device=device)


    def forward(self, X1, X2):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        # feature * batch_size
        output1 = self.model1(X1)
        output2 = self.model2(X2)

        return output1, output2










    #%%

class DecorrelatedNet(Network):
    """
    A convolutional network that allows to use (weighted) sums of probabilities
    or scores
    """

    def __init__(self, input_shape, signals_list, loss_coef, cca_type, plot_conflict=False):
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
             - loss_coef (float): the coefficient of the similarity loss
             - cca_type (string): either 'classical' or 'deep'
             - plot_conflict (bool): if True, compute the gradients of the
                 classification loss and decorrelation loss deparately, to show
                 the disagreement between them at the end of the training
                 default: False
           """
        super(DecorrelatedNet, self).__init__()
        self.loss_coef = loss_coef
        self.signals_list = signals_list
        self.networks = {}  # a dictionnary containing CNN as values
        self.cca_type = cca_type

        self.plot_conflict = plot_conflict
        if self.plot_conflict:
            self.losses_norms = {signal:{'classification':[], 'decorrelation':[]} for signal in signals_list}
            self.cosine_sim   = {signal:[] for signal in signals_list}

        for signal in signals_list:
            specific_CNN = CNN(input_shape, n_branches=1, fusion="start")
            specific_CNN.FC1 = nn.Identity() # effectively remove the last FC
                # layer without having to change the specific_CNN.forward() method

            self.networks[signal] = specific_CNN
            net_name = signal + "_CNN"
            self.add_module(net_name, specific_CNN)

        self.FC1 = nn.Linear(len(signals_list)*128, 8)

        # creation of the list of signals to consider
        self.couple_signals = []
        for i1, signal1 in enumerate(signals_list):
            for i2, signal2 in enumerate(signals_list[:i1]):
                self.couple_signals.append((signal1, signal2))


        if self.cca_type == 'classic':
            self.mean = {}           # will contain signals (strings) as keys
            self.base_change = {}    # will contain couple of signals as keys
            # aligned_feature = (feature - mean)*base_change

        elif self.cca_type == 'deep':
            self.correlators = {(signal1, signal2): Correlator(1600)
                             for signal1, signal2 in self.couple_signals}

        else:
            raise ValueError(f"unknown cca type '{cca_type}'\nMust be either 'classic' or 'deep'")

        self.correlations_history = {(signal1, signal2):[] for (signal1, signal2) in self.couple_signals}
            # for vizualisation purposes, store the
            #correlations each time a new CCA is performed



    def to(self, device):
        """
        We overridte the torch.nn.Module.to method so that the deep CCA models
        are sent to the GPU even if they are not in self.children()

        This method has the same signature as the overriden  method
        """
        if self.cca_type == 'classic':
            for     mean    in        self.mean.values():        mean.to(device)
            for base_change in self.base_change.values(): base_change.to(device)

        elif self.cca_type == 'deep':
            for correlator in self.correlators.values():
                correlator.to(device)
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
        dissimilar_features_dict (dict): keys are signal names, values are (batch_size, 128)
            tensors
        """

        classification_features_dict = {}
        dissimilar_features_dict = {}

        # the self.signals_list can contain several networks using the same input
        # eg Acc_norm_copy_0, Acc_norm_copy_1
        for signal in self.signals_list:
            if '_copy' in signal: # cut everything after the '_copy'
                _copy_index = signal.find('_copy')
                root_signal = signal[:_copy_index]
            else:
                root_signal = signal

            network = self.networks[signal]
#            features_dict[signal] = network(X[root_signal])

            X_specific = network.conv0(X[root_signal])
            X_specific = network.mp(network.relu_conv0(X_specific))
            X_specific = network.conv1(X_specific)
            X_specific = network.mp(network.relu_conv1(X_specific))
            X_specific = network.conv2(X_specific)
            X_specific = network.mp(network.relu_conv1(X_specific))
            X_specific = X_specific.view(X_specific.shape[0],-1)

            # fork: one X for dissimilarity lss without dropout, and one
            # X for classification, possibly using dropout
            X_classification = network.dropout0(X_specific)
            X_classification = network.relu_FC0(network.FC0(X_classification))
            X_classification = network.dropout1(X_classification)
            classification_features_dict[signal] = X_classification

#            X_dissimilarity = network.relu(network.FC0(X_specific))
            X_dissimilarity = X_specific
            dissimilar_features_dict[signal] = X_dissimilarity

        features = torch.cat([classification_features_dict[signal] for signal in self.signals_list], dim=1)

        scores = self.FC1(features)

        return scores, dissimilar_features_dict



    def predict(self, X):
        """
        Same as forward, but returns a class instead of scores
        The input, X, can be either a dict or an array, depending on the type of network and fusion

        Outputs:
            scores: a (batch_size, class_number) torch FloatTensor
            classes: a (batch_size, )            torch LongTensor
            dissimilar_features_dict: dict of (batch_size, 128) torch.FloatTensor
        """
        scores, dissimilar_features_dict = self(X)
        probas = F.softmax(scores, dim=1)
        classes = torch.argmax(probas, dim=1)
        return scores, classes, dissimilar_features_dict



    def get_gradients(self):
        """
        Record the gradients of the current loss and puts them in a dictionnary
        This function is used to

        Parameters: none

        Returns:
            gradients: dict
                keys = signal names (eg 'Acc_norm')
                values = 1-dimentional torch tensor: the flattened gradient
        """

        result_gradients = {}
        for signal in self.signals_list:
            net = self.networks[signal]

            list_gradients = []
            for w in net.parameters():
                list_gradients.append(w.grad.view(-1).clone())

            grad_vector = torch.cat(list_gradients, dim=0)
            result_gradients[signal] = grad_vector

        return result_gradients

    def record_losses_conflict(self, X,Y):
        """
        Computes the cosine similarity and norms of the decorrelation and
        classification losses, and put these values in two dictionnaries,
        self.losses_norms and self.cosine_sim

        Parameters
        ----------
        X : a dictionnary of (B, F, T) input tensors. The keys of this
            dictionnary need to be the same as the keys of self.networks
        Y : a torch tensor of (B, ) Long integers

        Returns
            None

        """
        _, _, dissimilar_features_dict = self.predict(X)
        decorrelation_loss = self.compute_decorrelation_loss(dissimilar_features_dict)
        decorrelation_loss.backward()
        grads_decorrelation = self.get_gradients()
        for p in self.parameters(): p.grad = None  # erase gradients

        scores, _, _ = self.predict(X)
        CE_loss = self.criterion(scores, Y)  # loss of the current batch
        CE_loss.backward()
        grads_classification = self.get_gradients()
        for p in self.parameters(): p.grad = None  # erase gradients


        for signal in self.signals_list:
            l_class_norm = torch.norm(grads_classification[signal])
            self.losses_norms[signal]['classification'].append(l_class_norm)
            l_decor_norm = torch.norm(grads_decorrelation[signal])
            self.losses_norms[signal]['decorrelation'].append(l_decor_norm)

            dot_product = torch.dot(grads_decorrelation[signal], grads_classification[signal])
            self.cosine_sim[signal].append(dot_product/(l_class_norm*l_decor_norm))


    def compute_decorrelation_loss(self, features_dict):
        """
        Parameters
        ----------
        features_dict: keys: signals (strings)
                       values: Pytorch tensors with shape (batch, n_features)

        Returns
        -------
        dissimilarity loss: Pytorch FloatTensor with size 1,
        """
        if self.cca_type == 'classic':
            decorrelation_loss = 0
            for (signal1, signal2) in self.couple_signals:
                aligned_features_1 = torch.matmul(features_dict[signal1] -self.mean[signal1], self.base_change[(signal1, signal2)])
                aligned_features_2 = torch.matmul(features_dict[signal2] -self.mean[signal2], self.base_change[(signal2, signal1)])
                decorrelation_loss -= nn.L1Loss()(aligned_features_1, aligned_features_2)

        elif self.cca_type == 'deep':
            decorrelation_loss = 0
            for (signal1, signal2) in self.couple_signals:
                aligned_features =  self.correlators[(signal1, signal2)](features_dict[signal1],
                                                                         features_dict[signal2])
                aligned_features_1, aligned_features_2 = aligned_features
                #decorrelation_loss -= nn.L1Loss()(aligned_features_1, aligned_features_2)
                decorrelation_loss = - max(nn.L1Loss()(aligned_features_1, aligned_features_2), nn.L1Loss()(aligned_features_1, -aligned_features_2))

        return decorrelation_loss



    def run_one_epoch(self, dataloader, gradient_descent):
        """
        Overrides the base_network.run_one_epoch to add the dissimilarity loss
        and the recalibration of the CCA.
        Browse one dataloader, and make predictions for each batch in it.
        Depending on the value of the gradient_descent variable, the weights of
        the model can also be trained

        Parameters
        ----------
        dataloader: a Pytorch DataLoader object, with n_instances, yielding (X, Y) couples. The
            dataloader can be empty, in which case a F1 score and a loss of -1
            are returned.
        gradient_descent (bool): true if we are training, false if we are
            only predicting

        Returns
        -------
        average_loss: average loss over the n_instances batches, if the dataloader is nonempty;
                1 otherwise
        f1_score: float, between 0. and 1. if the dataloader is nonempty, -1 otherwise
        predictions: array of ints (shape: (n_instances,))
        scores: array of floats (shape: (n_instances,8))
        ground_truth: array of ints corresponding to the objective. We
            return this list because Dataloaders can shuffle the data (shape: (n_instances,))
        """
        debug = self.debug

        predictions_list = []
        scores_list = []
        ground_truth_list = []
        losses_list = []
        self.train(gradient_descent)  # do not use dropout during train mode
        if gradient_descent:
            self.adapt_CCA(dataloader)


        for i_batch, (X, Y) in enumerate(dataloader):

            # if the model is ensemble of models, X is a dictionnary of tensors
            if isinstance(X, dict):
                for t in X.values(): t.requires_grad = gradient_descent
            else: # X is a tensor
                X.requires_grad = gradient_descent

            scores, predictions, dissimilar_features_dict = self.predict(X)

            CE_loss = self.criterion(scores, Y)  # loss of the current batch
            decorrelation_loss = self.compute_decorrelation_loss(dissimilar_features_dict)


            n_couples =len(self.couple_signals)

            loss = CE_loss + (self.loss_coef/n_couples) * decorrelation_loss

            losses_list.append(loss.detach())
            scores_list.append(scores.detach())
            predictions_list.append(predictions)

            ground_truth_list.append(Y)

            if gradient_descent:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            else:
                loss = loss.detach()

            if gradient_descent:
                print(i_batch)
                self.adapt_CCA(dataloader)




            if self.plot_conflict and gradient_descent: # plot on the train set only
                self.record_losses_conflict(X,Y)




        if len(dataloader)>0: # if the dataloader was nonempty:
            losses_list = [loss.item() for loss in losses_list]
            predictions  = torch.cat([pred  for pred  in  predictions_list], dim=0).to(torch.device('cpu')).detach().numpy()
            ground_truth = torch.cat([ gt   for  gt   in ground_truth_list], dim=0).to(torch.device('cpu')).detach().numpy()
            scores =       torch.cat([score for score in    scores_list   ], dim=0).to(torch.device('cpu')).detach().numpy()

            del ground_truth_list, scores_list, predictions_list
            # we fetch all data from the GPU at once, to minimize the time the GPU and CPU spend waiting each other.

            average_loss = np.mean(losses_list)  # average over the batches

            f1_value = f1_score(ground_truth, predictions, average='macro')

            # mode = 'train' if gradient_descent else 'val'
            # print("{}: loss = {:.3f}, F1_score = {:.3f}%".format(mode, average_loss, 100*f1_value))


        else: # the dataloader was empty
            average_loss, f1_value = -1, -1
            predictions, scores, ground_truth = np.array([]), np.array([]), np.array([])

        if debug:
            print('cuda.memory_allocated = %.2f Go' % (torch.cuda.memory_allocated()/10**9))

        return average_loss, f1_value, predictions, scores,  ground_truth




    def plot_CCA_components(self, dataloader, n_components):

        for signal1, signal2 in self.couple_signals:
            plt.figure()

            # subplots organization, legends, etc.
            subplot_handles = []
            h = ceil(np.sqrt(n_components))  # height (number of figures)
            w = ceil(n_components/h)         # width  (number of figures)
            for i_subplot in range(n_components):
               ax = plt.subplot(h, w, i_subplot+1)
               plt.title(str(i_subplot))
               if i_subplot // h == h-1: plt.xlabel(signal1)  # last line
               if i_subplot  % w == 0:   plt.ylabel(signal2)  # first column
               #plt.tick_params(axis='x', which='both', bottom=False, top=False,  labelbottom=False)  # cf https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
               #plt.tick_params(axis='y', which='both', right=False,  left=False, labelleft=False)
               subplot_handles.append(ax)

            legend_markers = []
            for i_class in range(len(classes_names)):
                legend_markers.append(Line2D([0], [0], marker='o', color=color_classes[i_class],
                                       lw=0, label='Scatter', markersize=5))
            plt.legend(legend_markers, classes_names) # we are currently using the last subplot


            # Feature computation and display
            for i_batch, (X, Y) in enumerate(dataloader):
                for t in X.values(): t.requires_grad = False

                scores, predictions, dissimilar_features_dict = self.predict(X)
                aligned_features = {}
                aligned_features[signal1] = torch.matmul(dissimilar_features_dict[signal1] -self.mean[signal1], self.base_change[signal1][signal2])
                aligned_features[signal2] = torch.matmul(dissimilar_features_dict[signal2] -self.mean[signal2], self.base_change[signal2][signal1])


                for i_component in range(n_components):
                    X_features = aligned_features[self.signals_list[0]][:,i_component].clone().detach().cpu().numpy()
                    Y_features = aligned_features[self.signals_list[1]][:,i_component].clone().detach().cpu().numpy()
                    colors = [color_classes[Y[i_class].item()]+[0.5] for i_class in range(Y.shape[0])]  # add 0.5 alpha
                    ax = subplot_handles[i_component]
                    ax.scatter(X_features, Y_features, s=1, c=colors)







    def adapt_CCA(self, dataloader):
        was_training = self.training
        self.train(False)


        if self.cca_type == 'classic':
            feature_blocks = {signal:np.zeros((len(dataloader.dataset), 1600)) for signal in self.signals_list}

            i_start = 0
            for (X, _) in dataloader:
                X_example = list(X.values())[0]
                i_end = i_start + X_example.shape[0] # add the batch_size of the current batch

                for X_tensor in X.values(): X_tensor.requires_grad = False
                _, dissimilar_features_dict = self(X)

                for signal in self.signals_list:
                    feature_blocks[signal][i_start:i_end,:] = dissimilar_features_dict[signal].clone().cpu().detach().numpy().reshape(X_example.shape[0], -1)

                #if i_start == 0:  print('X[3,0,3,3] (cca)', X['Acc_norm'][3,0,3,3].item(), dissimilar_features_dict['Acc_norm'][3,5].item(), dissimilar_features_dict['Acc_norm'].std().item())
                i_start = i_end



            for signal1, signal2 in self.couple_signals:
                X1_PCA = PCA(n_components=0.9999)
                X1_reduced = X1_PCA.fit_transform(feature_blocks[signal1])

                X2_PCA = PCA(n_components=0.9999)
                X2_reduced = X2_PCA.fit_transform(feature_blocks[signal2])

                n_components = min(X1_reduced.shape[1], X2_reduced.shape[1])
                if n_components == 1: print("only one feature left") ; return

                # this function uses the convention X.shape = (n_features, n_samples), which is the inverse of ours
                cca_results = cca_core.get_cca_similarity(X1_reduced.T, X2_reduced.T, verbose=False,
                                                          epsilon=1e-10, threshold=1-1e-6)

                # if both feature matrices have a different shape,  get_cca_similarity keeps the
                # features of the biggest one. We do not want them
                cca_results["coef_x"] = cca_results["coef_x"][:n_components,:]
                cca_results["coef_y"] = cca_results["coef_y"][:n_components,:]

                X1_transformed = X1_reduced @ (cca_results["coef_x"] @ cca_results["full_invsqrt_xx"]).T  # shape: (n_samples, n_features)
                X2_transformed = X2_reduced @ (cca_results["coef_y"] @ cca_results["full_invsqrt_yy"]).T  # shape: (n_samples, n_features)

                # sanity check: make sure we know to get back to the components
                self.X1_transformed = X1_transformed
                self.X2_transformed = X2_transformed

                correlations = np.diag(np.corrcoef(X1_transformed, X2_transformed, rowvar=False)[n_components:,:n_components])
                self.correlations_history[(signal1, signal2)].append(correlations)

                # We divide each components by its STD so that the variance of new components is 1
                # This allows to keep the loss in check (ie, it does not increase nor decrease too much)
                self.mean[signal1] = X1_PCA.mean_.reshape(1, -1)
                self.base_change[(signal1,signal2)] =  X1_PCA.components_.T @ cca_results["full_invsqrt_xx"].T @ cca_results["coef_x"].T  /np.std(X1_transformed, axis=0, keepdims=True)
                self.mean[signal2] = X2_PCA.mean_.reshape(1, -1)
                self.base_change[(signal2,signal1)] =  X2_PCA.components_.T @ cca_results["full_invsqrt_yy"].T @ cca_results["coef_y"].T /np.std(X2_transformed, axis=0, keepdims=True)


                self.mean[signal1]          = torch.tensor(self.mean[signal1],          device=device)
                self.base_change[(signal1,signal2)] = torch.tensor(self.base_change[(signal1,signal2)], device=device)
                self.mean[signal2]          = torch.tensor(self.mean[signal2],          device=device)
                self.base_change[(signal2,signal1)] = torch.tensor(self.base_change[(signal2,signal1)], device=device)


        elif self.cca_type == 'deep':
            for signal1, signal2 in self.couple_signals:
                correlator = self.correlators[(signal1, signal2)]
                correlator.optim.zero_grad()

                for X,_ in dataloader: # we will only go through one batch
                    _, _, dissimilar_features_dict = self.predict(X)
                    aligned_features = correlator(dissimilar_features_dict[signal1],
                                                  dissimilar_features_dict[signal2]) # couple of torch tensors

                    decorrelation_loss = -correlator.criterion.loss(aligned_features[0], aligned_features[1])

                    for net in self.networks.values(): net.zero_grad()
                    decorrelation_loss.backward()
                    correlator.optim.step()
                    for net in self.networks.values(): net.zero_grad()

                    break

        self.train(was_training)



    def plot_correlations(self):
        for signal1, signal2 in self.couple_signals:
            max_n_elements = max([corr_array.shape[0] for corr_array in self.correlations_history])
            img_array = np.zeros((len(self.correlations_history[(signal1, signal2)]), max_n_elements)) -1
            for i, corr_array in enumerate(self.correlations_history):
                img_array[i,:corr_array.shape[0]] = corr_array

            plt.figure()
            plt.pcolormesh(img_array.T, vmin=-1, vmax=1, cmap="seismic")
            plt.title(f"correlations after each CCA update\ndissimilarity weight={self.loss_coef}")
            plt.colorbar()
            plt.show()



    def plot_losses_conflict(self):

        n_signals = len(self.signals_list)
        plt.figure()

        for i_signal, signal in enumerate(self.signals_list):
            plt.subplot(2,n_signals+1,i_signal +1)
            plt.title(signal)
            plt.plot([l.item() for l in self.losses_norms[signal]['classification']], 'o', markersize=1)
            plt.plot([l.item() for l in self.losses_norms[signal]['decorrelation']], 'o', markersize=1)
            plt.legend(['classification', f'decorrelation \n(coeff = {self.loss_coef:.1e})'])
            if i_signal == 0: plt.ylabel('gradient norms')

            plt.subplot(2,n_signals+1,i_signal +n_signals+1 +1)
            plt.plot([l.item() for l in self.cosine_sim[signal]], 'go', markersize=1)
            plt.ylim(-1, 1)
            plt.xlabel('loss evaluation index')
            if i_signal == 0: plt.ylabel('scalar product')



        # total loss
        list_class_norms, list_decor_norms = [], []
        list_cosine_sim = []

        n_losses = len(self.losses_norms[signal]['classification'])
        for i in range(n_losses):
            #norm: Pythoagoras theorem
            class_norms_list = [self.losses_norms[signal]['classification'][i] for signal in self.signals_list]
            class_norm = torch.sqrt(sum([norm**2 for norm in class_norms_list])).item()
            list_class_norms.append(class_norm)

            decor_norms_list = [self.losses_norms[signal]['decorrelation'][i] for signal in self.signals_list]
            decor_norm = torch.sqrt(sum([norm**2 for norm in decor_norms_list])).item()
            list_decor_norms.append(decor_norm)


            # scalar product: get back to the dot product
            dot_products_list = [self.cosine_sim[signal][i] *
                                (self.losses_norms[signal]['classification'][i] * self.losses_norms[signal]['decorrelation'][i])
                                        for signal in self.signals_list]
            dot_product = sum(dot_products_list).item()
            list_cosine_sim.append(dot_product/(class_norm * decor_norm))


        plt.subplot(2,n_signals+1,n_signals +1)
        plt.title("Complete network")
        plt.plot(list_class_norms, 'o', markersize=1)
        plt.plot(list_decor_norms, 'o', markersize=1)
        plt.legend(['classification', f'decorrelation \n(coeff = {self.loss_coef:.1e})'])

        plt.subplot(2,n_signals+1,2*n_signals +2)
        plt.plot(list_cosine_sim, 'go', markersize=1)
        plt.ylim(-1, 1)
        plt.xlabel('loss evaluation index')


        plt.show()


    #%%
if __name__ == "__main__":
    from preprocess import Datasets
    import torch.utils.data
    from preprocess.transforms import SpectrogramTransform
    from preprocess.fusion import separate_sensors_collate
    from param import fs, duration_window, duration_overlap, spectro_batch_size

    spectrogram_transform = SpectrogramTransform(["Acc_norm", "Gyr_y"], fs, duration_window, duration_overlap,
                                                 spectro_batch_size, interpolation='linear', log_power=True, out_size=(48,48))
    collate_fn = separate_sensors_collate

    try :  # do not reload the datasets if they already exist
        train_dataset
        val_dataset

    except NameError:
        train_dataset = Datasets.SignalsDataSet(mode='train', split='balanced', comp_preprocess_first=True, transform=spectrogram_transform)
        val_dataset =   Datasets.SignalsDataSet(mode='val',   split='balanced', comp_preprocess_first=True, transform=spectrogram_transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, num_workers=0, shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, collate_fn=collate_fn, num_workers=0)

    str_result = ""


    model = DecorrelatedNet(input_shape=(1,48,48), signals_list=["Acc_norm", "Gyr_y"], loss_coef=0.1,
                            plot_conflict=True, cca_type='deep')

    model.to(device)
    model.adapt_CCA(train_dataloader)

    _, _, _, val_F1 = model.train_process(train_dataloader, val_dataloader, maxepochs=10)
    model.adapt_CCA(train_dataloader)





