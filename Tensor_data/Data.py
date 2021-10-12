import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TTP_dataset(Dataset):

    def __init__(self, num_nodes, x, capacity):
        super(TTP_dataset, self).__init__()

        self.dataset = torch.from_numpy(x).float()
        self.dynamic = torch.zeros(1, 1, num_nodes)
        self.num_nodes = num_nodes
        self.size = 1
        self.capacity =capacity
       


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
         return (self.dataset[idx], self.dynamic[idx], [])