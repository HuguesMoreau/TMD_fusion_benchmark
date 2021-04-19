"""
Author Hugues

Create a base class all our networks will inherit from
Use it to define universal parameters
"""

from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

from param import classes_names, device
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
print("working on: ",device)

import numpy as np

class Network(nn.Module):
    def __init__(self, loss_weights=None, debug=0):
        """
        The base class for all networks.
        Here we define training procedure and the loss
        """
        super(Network, self).__init__()

        if loss_weights == None:
            class_hist = [2302, 2190,  686, 2101, 2475, 2083, 2520, 1953]  # from visualiztions/classes.py
            loss_weights = [1/p for p in class_hist]


        loss_weights = torch.FloatTensor(loss_weights).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=loss_weights)

        self.debug=debug


        # history of losses and F1 scores for plots
        self.losses_train = []
        self.losses_val   = []
        self.F1_train = []
        self.F1_val   = []



    def forward(self, X):
        raise NotImplementedError("The forward method must be implemented in every network inheriting from this class")



    def predict(self, X):
        """
        Same as forward, but returns a class instead of scores
        The input, X, can be either a dict or an array, depending on the type of network and fusion

        Outputs:
            scores: a (batch_size, class_number) torch FloatTensor
            classes: a (batch_size, )            torch LongTensor
        """
        scores = self(X)
        probas = F.softmax(scores, dim=1)
        classes = torch.argmax(probas, dim=1)
        return scores, classes



    def run_one_epoch(self, dataloader, gradient_descent):
        """
        Browse one dataloader, and make predictions for each batch in it.
        Depending on the value of the gradient_descent variable, dropout may or
        may not be applied

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

        for i_batch, (X, Y) in enumerate(dataloader):
            # if the model is ensemble of models, X is a dictionnary of tensors
            if isinstance(X, dict):
                for t in X.values(): t.requires_grad = gradient_descent
            else: # X is a tensor
                X.requires_grad = gradient_descent

            scores, predictions = self.predict(X)

            loss = self.criterion(scores, Y)  # loss of the current batch
            losses_list.append(loss.detach())

            scores_list.append(scores.detach())
            predictions_list.append(predictions)

            ground_truth_list.append(Y)

            if gradient_descent:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            else:
                loss = loss.detach()


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





    def train_process(self, train_dataloader, val_dataloader, maxepochs):
        """
        The base training process.

        Parameters
        ----------
        train_dataloader (pytorch DataLoader object)
        val_dataloader (pytorch DataLoader object)
        maxepochs (int)

        Returns
        -------
        The scores at the last epoch: a 4-tuple of:
        train_loss (positive float)
        val_loss (positive float)
        train_f1 (float, between 0. and 1.)
        val_f1 (float, between 0. and 1.)
        """

        if maxepochs > 1: print("\nTraining the model:")
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)
            # optim is defined here because it needs to access to the model's
            # parameters() method, which requires the architecture to be fixed


        for epoch in range(maxepochs):
            # print("\n epoch {}".format(epoch))
            train_loss, train_f1, _, _, _ = self.run_one_epoch(train_dataloader, gradient_descent=True)
            with torch.no_grad():
                val_loss,   val_f1,   _, _, _ = self.run_one_epoch(val_dataloader,   gradient_descent=False)

            self.losses_train.append(train_loss)
            self.losses_val.append(val_loss)
            self.F1_train.append(train_f1)
            self.F1_val.append(val_f1)

            if maxepochs > 1:
                print("epoch %4d/%4d " % (epoch,maxepochs-1),
                      ' train_loss: %.4f' % train_loss, ' val_loss: %.4f' % val_loss,
                      ' val F1: %.4f' % val_f1)

        if maxepochs == 0: train_loss, val_loss, train_f1, val_f1 = -1,-1,-1,-1

        return train_loss, val_loss, train_f1, val_f1



    def test(self, train_dataloader, val_dataloader, test_dataloader, maxepochs):
        """
        trains the model on {train + val}, and tests it on {test} once

        Parameters
        ----------
        train_dataloader (pytorch DataLoader object)
        val_dataloader (pytorch DataLoader object)
        test_dataloader (pytorch DataLoader object)
        maxepochs (int)

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

        for epoch in range(maxepochs):
            # print("\n epoch {}".format(epoch))
            train_loss, train_f1, _, _, _ = self.run_one_epoch(train_dataloader, gradient_descent=True)
            val_loss,   val_f1,   _, _, _ = self.run_one_epoch(val_dataloader,   gradient_descent=True)

            print("epoch %4d/%4d " % (epoch,maxepochs-1),
                  ' train_loss: %.4f' % train_loss, ' val_loss: %.4f' % val_loss,
                  ' val F1: %.4f' % val_f1)

        with torch.no_grad():
            test_loss, test_f1, _, _, _ = self.run_one_epoch(test_dataloader, gradient_descent=False)

        print("at the end of training, ",
                  ' test_loss: %.4f' % test_loss, ' test_f1: %.4f' % test_f1 )

        return train_loss, val_loss, test_loss, train_f1, val_f1, test_f1


    #%% vizualizations

    def plot_learning_curves(self, title=""):
        """
        Displays the F1 vs number of epochs and loss vs number of epochs
        """
        fontdict = {'fontsize':7}

        plt.figure()
        plt.plot(self.F1_train)
        plt.plot(self.F1_val)
        plt.title(title + "\nF1-score", fontdict=fontdict)
        plt.legend(["train", "val"])
        plt.grid()
        plt.show()



        plt.figure()
        plt.plot(self.losses_train)
        plt.plot(self.losses_val)
        plt.title(title + "\nLoss", fontdict=fontdict)
        plt.legend(["train", "val"])
        plt.grid()
        plt.show()



    def plot_confusion_matrix(self, val_dataloader, title=""):
        with torch.no_grad():
            _, f1_value, predictions, _, ground_truth = self.run_one_epoch(val_dataloader,   gradient_descent=False)

        plt.figure()
        M = confusion_matrix(ground_truth, predictions)
        n = M.shape[0]

        sum_samples = np.tile(np.sum(M, axis=1).reshape(n,1), (1,n))
        M = 100*M/sum_samples

        fig, ax = plt.subplots(nrows=1)

        im = ax.imshow(M)
        cbar = ax.figure.colorbar(im, ax=ax)#, **cbar_kw)
        cbar.ax.set_ylabel("Recall", rotation=-90, va="bottom", size=15)

        ax.set_xticklabels([""]+classes_names, size=10)
        ax.set_yticklabels([""]+classes_names, size=10)  # this function ignires the firts element of the list

        for i in range(n):
            for j in range(n):
                colour = "w" if i!= j else "k"
                ax.text(j, i, np.round(M[i, j],1),
                               ha="center", va="center", color=colour, size=10)

        ax.set_xlabel("Predicted", size=10)
        ax.xaxis.set_label_position('bottom')
        ax.set_ylabel("Ground Truth", size=10)

        ax.set_title(title + "\n final F1: {:.1f}".format(f1_value*100))
        plt.show()



