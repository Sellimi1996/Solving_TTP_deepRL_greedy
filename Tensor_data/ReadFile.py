import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_file(filename):
    file = filename
    data=[]
    variables = {}
    with open(file,'r') as f: 
            lines = f.readlines()
            lines = lines[0:]
            lines = [x.strip() for x in lines]
            lines = [x.replace(':','') for x in lines]  # removes all semi colon
            lines = [x.replace('(','') for x in lines]  # removes all (
            lines = [x.replace(')','') for x in lines]  # removes all )
            lines = [x.split('\t') for x in lines]

    for i in range(len(lines)):
        
        # starting of the reading of coords
        if lines[i][0] == "NODE_COORD_SECTION": 
            start=i
        
        # Number of data points
        if lines[i][0] == "DIMENSION":
            dimension=int(lines[i][1])
            
        # Number of total items
        if lines[i][0] == "NUMBER OF ITEMS ":
            items=int(lines[i][1])
            
        # Knapsack capacity
        if lines[i][0] == "CAPACITY OF KNAPSACK ":
            weight_limit=int(lines[i][1])
            
        # Minimum velocity
        if lines[i][0] == "MIN SPEED ":
            vmin = float(lines[i][1])   
            
        # Maximum velocity
        if lines[i][0] == "MAX SPEED ":
            vmax = float(lines[i][1])
            
        # Rent ratio
        if lines[i][0] == "RENTING RATIO ":
            rent_ratio = float(lines[i][1])
        # name instances
        if lines[i][0] == "PROBLEM NAME ": 
            dataName = str(lines[i][1])
        # Knapsack type    
        if lines[i][0] == "KNAPSACK DATA TYPE ": 
            TypKnap = str(lines[i][1])
    num_nodes = dimension        
    loc= np.array(lines[start+1:start+dimension+1], dtype=float)
    loc = loc[np.argsort(loc[:,0])]
    start_loc = [loc[0,1],loc[0,2]]
       # all X coords
    x_loc = loc[:,1]
       # all Y coords
    y_loc = loc[:,2]

      # reading item list
    bag = np.array(lines[start+dimension+2 : start+dimension+items+2], dtype=float)    
      # sorting according to allocated cities (last column)
    bag = bag[np.argsort(bag[:,-1])]
    
      # Profit array for each items after sorting
      # profit[0:item_per_city] is profit for items from 2nd city
    profit = bag[:,1]
    profit = profit / (np.max(profit,0))
      # weights array for each items after sorting
      # weights[0:item_per_city] is weights for items from 2nd city
   
    weights = bag[:,2]
    tspwgh=np.zeros(num_nodes)
    tspwgh[0]=0.
    i=1
    j=0
    while j < items and i < num_nodes:
        tspwgh[i]=weights[j]
        i=i+1
        j=j+1
    tspwgh= np.array(tspwgh)
    capacity=weight_limit/(np.max(weights))
    weights = weights / (np.max(weights,0))
      # node array for each items after sorting
      # node[0:item_per_city] is node-2 == [2,2,2,2,2]
    node = bag[:,3]
    
      # combining location and item data
    loc_bag = np.insert(bag,[np.ma.size(bag,1)],[0,0],axis=1)
    
    df_loc_bag = pd.DataFrame(loc_bag,columns=[lines[start+dimension+1][1].split(',')+lines[start][1].split(',')[1:]])
    
    #bag_sort = df_bag.sort_values(by=[' ASSIGNED NODE NUMBERs'])
    for i in range(len(df_loc_bag)):
        a = df_loc_bag.loc[i].at[' ASSIGNED NODE NUMBER'].item()
        df_loc_bag.loc[i].at[' X'] = loc[np.where(loc[:,0]==a)[0][0],1]
        df_loc_bag.loc[i].at[' Y'] = loc[np.where(loc[:,0]==a)[0][0],2]
    
    
     # Normalize the data in the tour coords
    TSPxy = np.transpose(np.array([x_loc,y_loc]))
    SubProblemTSP = np.transpose(np.array([x_loc,y_loc,tspwgh]))
    SubProblemTSP = SubProblemTSP / (np.max(SubProblemTSP,0))
    SubProblemTSP = SubProblemTSP.T
    
     # Normalize the data in the Packing Plan
     
    SubProblemKP = np.transpose(np.array([profit,weights,node]))
    SubProblemKP = SubProblemKP.T
    
    
    SubProblemTSP = SubProblemTSP.reshape(1, 3, num_nodes)
    SubProblemKP = SubProblemKP.reshape(1, 3, items)
    
    return SubProblemTSP,TSPxy,SubProblemKP, vmax, vmin, rent_ratio, weight_limit, capacity, items, num_nodes, df_loc_bag, dataName,TypKnap
