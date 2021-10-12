import numpy as np
from TspandKpindependent.LightKP import bestSac
def getPackedItem(Index_items,df, capacity, tour, tspdata, num_nodes, R):
    profit=[]
    weight=[]
    k=len(Index_items)
    # fill the profit and weight according to list of items (index_items) belongs to the given tour
    for i in range(len(Index_items)):
        for j in range(len(df)):
            if Index_items[i]== df.loc[j].at['INDEX'].item():
                profit.append(df.loc[j].at[' PROFIT'].item())
                weight.append(df.loc[j].at[' WEIGHT'].item())
    Sac, reward = bestSac(tour, df, num_nodes, Index_items, R, capacity, tspdata, profit, weight, k)
    optimal_Sac = Sac
    

    return optimal_Sac