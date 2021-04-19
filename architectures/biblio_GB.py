"""
Author Hugues


A third-party (unofficial) implementation of :
Wang, Weiyao, Du Tran, et Matt Feiszli.
« What Makes Training Multi-Modal Networks Hard? »
arXiv:1905.12681 [cs], May 29, 2019. http://arxiv.org/abs/1905.12681.

"""

if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")


import torch

from architectures.base_network import Network
from architectures.late_fusion import LateFusionCNN




class GBlend_CNN(LateFusionCNN):
    def __init__(self, input_shape, signals_list):
        """
        Inputs :
            - input_shape (triple of ints, (channel, freq, time))
                If there is only one input channel, the next two arguments
                (n_signals and fusion) are ignored
            - signals_list (list of strings): Each string is the name of one
                network. The names need to correspond to the names given to the
                transform funtion (ex: 'Acc_norm', 'Gyr_y')
           """
        LateFusionCNN.__init__(self, input_shape, signals_list, fusion_mode='probas', use_weights=False)
        # a sum of losses corresponds to a sum of probas, where the weights of the
        # sum of probas are the exponential of the weights of the sum of losses
        # as the loss we use is:  loss = -log(p_gt)
        # so   w*loss = log(exp(w)) * (-log(p_gt))  = -log(p_gt * exp(w))
        # reminder : setting use_weights to False creates weights that are equal
        # to 1/M, and constant (not learned). We will replace these values with
        # the actual coefficients in the training_process function




    def get_weights(self, signals_list=None, dataloader=None):
        """
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
        weights: a dictionnary of floats
        """
        if signals_list==None: signals_list = self.signals_list

        coefficients = {}
        for signal in signals_list:
            coefficients[signal] = self.weight_layers[signal].weight.data.item()
        return coefficients






    def train_process(self, train_dataloader, val_dataloader, maxepochs, proportion_separate=0.5):
        """
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

        # ------------- buiding subsets of the training dataset ----------------
        train_dataset = train_dataloader.dataset
            # it may seem weird to extract the dataset instead of asking it as an argument,
            # but doing as we do allows the function to have the same signature as LateFusionCNN's
        index_train_0 = [i for i in range(len(train_dataset)) if i <  0.8*len(train_dataset) ]
        index_train_1 = [i for i in range(len(train_dataset)) if i >= 0.8*len(train_dataset) ]
        index_train_0 = list(range(len(train_dataset)))
        index_train_1 = list(range(len(train_dataset)))

        train_0_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataloader.batch_size,
                          collate_fn=train_dataloader.collate_fn,
                          sampler=torch.utils.data.SubsetRandomSampler(index_train_0))
        train_1_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataloader.batch_size,
                          collate_fn=train_dataloader.collate_fn,
                          sampler=torch.utils.data.SubsetRandomSampler(index_train_1))
        losses_final_train, losses_final_val, losses_zero_train, losses_zero_val = {}, {}, {}, {}



        # ----------------------------------- training --------------------------------------
        if maxepochs_separate == 0 :
           # raise ValueError("We need at least one epoch of separate training")
            pass
        else:
            global_collate_fn = train_dataloader.collate_fn
            for signal in self.signals_list:
                print('\n\n', signal)

                if '_copy' in signal: # cut everything after the '_copy'
                    _copy_index = signal.find('_copy')
                    signal_root = signal[:_copy_index]
                else:
                    signal_root = signal

                # we *temporarily* replace the collate function of the dataloader
                def select_signal(L):
                    (X, Y) = global_collate_fn(L)
                    return X[signal_root], Y

                train_0_dataloader.collate_fn = select_signal
                train_1_dataloader.collate_fn = select_signal
                val_dataloader.collate_fn   = select_signal

                loss_zero_train = self.networks[signal].run_one_epoch(train_0_dataloader, gradient_descent=False)[0]
                loss_zero_val =   self.networks[signal].run_one_epoch(train_1_dataloader, gradient_descent=False)[0]
                losses_zero_train[signal] = loss_zero_train
                losses_zero_val[signal] =   loss_zero_val

                loss_final_train, loss_final_val, _, _ = Network.train_process(self.networks[signal], train_0_dataloader, train_1_dataloader, maxepochs_separate)
                losses_final_train[signal] = loss_final_train
                losses_final_val[signal] =   loss_final_val


            train_dataloader.collate_fn = global_collate_fn
            val_dataloader.collate_fn   = global_collate_fn

            # ---------------------------- Weight computation -------------------------------
            coefficient = {}
            print("\nsignal      L_0t    L_0v    L_ft    L_fv      O       G    weight")
            for signal in self.signals_list:
                O = (losses_zero_train[signal] - losses_zero_val[signal]) - (losses_final_train[signal] - losses_final_val[signal])
                G = losses_zero_val[signal] - losses_final_val[signal]
                coefficient[signal] = G/(O**2)
                str_to_print =  f'{signal:8s}    ' + \
                       f'{losses_zero_train[signal]:.3f}   {losses_zero_val[signal]:.3f}   ' + \
                       f'{losses_final_train[signal]:.3f}   {losses_final_val[signal]:.3f}   ' + \
                       f'{O:.3f}   {G:.3f}   {coefficient[signal]:.3f}'
                print(str_to_print)
            print('\n\n')

            sum_coefficients = sum(coefficient.values())
            for signal in self.signals_list:
                coefficient[signal] /= sum_coefficients

            # as we set use_weights = False in the __init__ function of LateFusionCNN,
            # the weights will not move (they have require_grad = False)
            for signal in self.signals_list:
                self.weight_layers[signal].weight.data[0] = coefficient[signal]
                # Weighing the losses with w is the same as weighing the probas
                #    with exp(w) because we use NLLLoss: loss = -log(p_gt)
                #    w*loss = log(exp(w)) * (-log(p_gt))  = -log(p_gt * exp(w))
                # BUT: the weights go through a softmax before being averaged, (see
                #    the LateFusionCNN.forward() method), so we cancel the softmax
                #    with a log.



        print('Global model')
        if maxepochs_ensemble >0 :
            results = Network.train_process(self, train_dataloader, val_dataloader, maxepochs_ensemble)
        else:
            results = Network.train_process(self, [],               val_dataloader, 1)
            #the training loss and f1 will be equal to -1, but the validation loss and f1 will be computed (once)


        # print the final coefficients
        coefficients = [self.weight_layers[signal].weight.data for signal in self.signals_list]
        coefficients = torch.cat(coefficients)
        print(self.signals_list)
        print(torch.softmax(coefficients, dim=0).cpu().detach().numpy())

        return results






    #%%
if __name__ == "__main__":
    from preprocess import Datasets
    import torch.utils.data
    from preprocess.transforms import SpectrogramTransform
    from preprocess.fusion import separate_sensors_collate
    from param import fs, duration_window, duration_overlap, spectro_batch_size, device

    spectrogram_transform = SpectrogramTransform(["Acc_norm", "Gyr_y"], fs, duration_window, duration_overlap,
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


    model = GBlend_CNN(input_shape=(1,48,48), signals_list=["Acc_norm", "Gyr_y"])
    model.to(device)
    print(model.get_weights())
    model.train_process(train_dataloader, val_dataloader, maxepochs=50, proportion_separate=0.5)
    print(model.get_weights())




