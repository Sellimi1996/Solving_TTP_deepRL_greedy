"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import knapsack


class TSPDataset(Dataset):

    def __init__(self, size=50, num_samples=1e6, seed=None, k=49, capacity=1.8, R=0.71):
        super(TSPDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset = torch.rand((num_samples, 2, size))
        self.dynamic = torch.zeros(num_samples, 1, size)
        self.num_nodes = size
        self.size = num_samples
        self.num_items = k
        self.capacity = capacity
        self.R=R

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], self.dataset[idx, :, 0:1])


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask


def reward(static, tour_indices, num_nodes, kp_dataset, k, capacity, R):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """
    w_c=0
    vmax=1
    vmin=0.1
    v_c = vmax - w_c * (vmax - vmin) / capacity
    tour=[]
    profit_pickeditem=0
    # extrat from tour indices the list of items
    tour.append(tour_indices.cpu().numpy())
    item_list=[]
    for i in range(0, num_nodes):
        for j in range(0, k):
            if np.array(tour)[:,i] == kp_dataset[:,2,j]-1:
                index = j+1
                item_list.append(index)
    
    # found the best picked item with max profit 
    profit=[]
    weight=[]
    # fill the profit and weight according to list of items (index_items) belongs to the given tour
    for i in range(len(item_list)):
        for j in range(0,k):
            if item_list[i]== kp_dataset[:,2,j]:
                profit.append(kp_dataset[:,0,j].cpu().numpy())
                weight.append(kp_dataset[:,1,j].cpu().numpy())
    
    # applied dynamic programming to solve this given items 
    total_reward, choices = knapsack.knapsack(weight, profit).solve(capacity)
    
    # decode the result to represent a solution for knapsack 
    xs = np.zeros(len(item_list))
    for i in choices:
        xs[i] = 1
        
    packeditem = np.zeros(len(item_list))
    for i in range(0, len(item_list)):
        packeditem[i]= item_list[i]*xs[i]
    
    sac=[]
    for i in range(0, len(item_list)):
        if packeditem[i]!=0:
            sac.append(packeditem[i])
    # implement distance matrix
    dist_mat = np.zeros((num_nodes, num_nodes))
    xcoords = static[:, 0].clone()
    ycoords = static[:, 1].clone()
    obj1data = static.data.clone().detach()
    l = obj1data.size()[1]
    for i in range(0,l):
        for j in range(0,l):
            #dist_mat[i][j] = np.sqrt((np.array(obj1)[:,0,i]-np.array(obj1)[:,0,j])**2 + (np.array(obj1)[:,1,i]-np.array(obj1)[:,1,j])**2)
            dist_mat[i][j] = torch.sqrt(torch.sum(torch.pow(obj1data[:, i] - obj1data[:, j], 2))).cpu().numpy()
    
    w_c=0
    vmax=1
    vmin=0.1
    i=0
    v_c=vmax-w_c*(vmax-vmin)/capacity
    tes=0
    
    while i+1< len(tour):
            for j in range(0,k):
                if kp_dataset[:,2,j]== tour[i]+1:
                    index = j+1
                    for y in range(len(sac)):
                        if sac[y]==index:
                            w_c+=kp_dataset[:,1,j]
                            v_c=vmax-w_c*(vmax-vmin)/capacity
            tes+=dist_mat[tour[i]][tour[i+1]]/v_c
            i=i+1
    tes+=dist_mat[tour[len(tour)-1]][tour[0]]/v_c
    res = tes
        
    
    # Bi-TTP by traveling time and profit
    profit= total_reward
    profit_pickeditem += profit
    traveling_time = res.sum(1)
    
    obj = profit_pickeditem-traveling_time*R
    
    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    obj1= torch.tensor(traveling_time)
    obj2= torch.tensor(profit_pickeditem)

    return obj, obj1, obj2   


def render(static, tour_indices, save_path):
    """Plots the found tours."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)