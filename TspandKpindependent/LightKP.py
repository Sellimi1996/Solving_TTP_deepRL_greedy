from TspandKpindependent.distance import distance
from TspandKpindependent.ValueFunction import tourLength
import numpy as np
from TspandKpindependent.ScoreItem import getlight_Item
from operator import itemgetter
# initilisation parameter

def bestSac(tour, df, num_nodes, list_items, R, max_capacity, tspdata, p, w, k):
    # Compute the score of each item depend the distancs tour
    score_items = getlight_Item(tour, df, num_nodes, list_items, R, max_capacity, tspdata)
    # sorted the items depend score value in descendent order
    tab=np.transpose(np.array([list_items, score_items, w, p]))
    Sorted_tab = sorted(tab, key=itemgetter(1), reverse=True)
    bestreward=0
    tes=0
    totale_profit=0
    # distance matrix 
    dis = distance(tspdata, num_nodes)
    # parameter definetion
    item=[]
    selected_item=[]
    current_weight = 0
    # pick the top item score
    if Sorted_tab[0][2] < max_capacity:
        current_weight = Sorted_tab[0][2]
        item.append(Sorted_tab[0][0])
        selected_item.append(Sorted_tab[0][0])
        i=1
        y=0

    W=max_capacity
    dis = distance(tspdata, num_nodes)
    # calculate the TTP objective function used given tour and the top score item
    bestreward, tes, totale_profit= tourLength(tour,num_nodes, item, df, k, R, max_capacity, dis)
    # check the other item to maximize the rewards without violate the capacity of Knapsack
    while i < len(list_items):
        index = Sorted_tab[i][0]
        if current_weight+Sorted_tab[i][2]<W:
            item.append(index)
            y=y+1
            reward,  tes, totale_profit= tourLength(tour,num_nodes, item, df, k, R, max_capacity, dis)
            if reward > bestreward:
                selected_item.append(index)
                bestreward=reward
                current_weight+= Sorted_tab[i][2]
            else:
                item.pop(len(item)-1)
        i=i+1
    return selected_item, bestreward 
    
     
     