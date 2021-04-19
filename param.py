from pathlib import Path


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_path = Path(r"[data_path directory]")

"""Expected file tree :
[data_path directory]
    train
        Acc_x.txt
        Acc_y.txt
        Acc_z.txt
        Gra_x.txt
        etc.
    test
        Acc_x.txt
        etc.

"""


classes_names = ["Still","Walk","Run","Bike","Car","Bus","Train","Subway"]
# cf the last page of    Wang, Lin, Hristijan Gjoreskia, Kazuya Murao, Tsuyoshi Okita, and Daniel Roggen. « Summary of the Sussex-Huawei Locomotion-Transportation Recognition Challenge »
# https://doi.org/10.1145/3267305.3267519.


fs = 100 # sampling frequency

duration_window = 5     # parameters for spectrogram creation
duration_overlap = 4.9
duration_segment = 60

spectro_batch_size = 1000  # parameter for spectrogram computation.
  # We cannot create 13,000 spectrograms at once, so we compute them by batches
  # you may adapt this depending on your RAM



