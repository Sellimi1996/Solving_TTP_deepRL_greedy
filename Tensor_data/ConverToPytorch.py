import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
    
def convert_to_pytorch(x,y):
    dataset = torch.from_numpy(x).float()
    dynamic = torch.zeros(1, 1, y)
    num_nodes = y
    return dataset, dynamic, num_nodes
 