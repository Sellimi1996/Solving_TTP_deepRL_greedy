from TspandKpindependent.distance import distance
import numpy as np

def tourLength(tour, num_nodes, packedPlan, df, k, R, max_cap, data):
    Euc_matrix = distance(data, num_nodes)
    w_c=0 #current weight of knapsak
    vmax=1 # maximum velocity
    vmin=0.1 # minimum velocity
    tes=0
    i=0
    v_c=vmax-w_c*(vmax-vmin)/max_cap
    tes=0
    totale_profit_kp=0
    
    #compute traveling time by velocity v_c, which depend to current weight of knapsack
    while i+1 < num_nodes:   
            for j in range(0,k):
                if df.loc[j].at[' ASSIGNED NODE NUMBER'].item()== tour[0,i]+1:
                    index = df.loc[j].at['INDEX'].item()
                    for y in range(len(packedPlan)):
                        if packedPlan[y] == index:
                            w_c +=df.loc[j].at[' WEIGHT'].item()
                            v_c=vmax-w_c*(vmax-vmin)/max_cap
                            
            v_c=vmax-w_c*(vmax-vmin)/max_cap                      
            tes+=Euc_matrix[tour[0,i],tour[0,i+1]]/v_c
            i=i+1
        
        
        
    for j in range(0,k):
            if df.loc[j].at[' ASSIGNED NODE NUMBER'].item()== tour[0,num_nodes-1]+1:
                    index = df.loc[j].at['INDEX'].item()
                    for y in range(len(packedPlan)):
                        if packedPlan[y] == index:
                            w_c +=df.loc[j].at[' WEIGHT'].item()
                            v_c=vmax-w_c*(vmax-vmin)/max_cap   
    tes+= Euc_matrix[tour[0,num_nodes-1],tour[0,0]]/v_c
    
    
    
    
         # compute the real profit of knapsak
    for i in range(len(packedPlan)):
           for j in range(0,k):
                if df.loc[j].at['INDEX'].item()==packedPlan[i]:
                    totale_profit_kp = totale_profit_kp +df.loc[j].at[' PROFIT'].item()
    obj_func=0
    obj_func= totale_profit_kp -(tes*R)
                
    return obj_func,tes, totale_profit_kp

def getItem_score(tour, df, num_nodes, item_list, R, W, tspdata):
    Euc_matrix = distance(tspdata, num_nodes)
    items=np.zeros(len(item_list))
    ratio=0.9/W
    z=0
    i=1
    v_c=1
    while z<len(item_list):
        if  i+1<num_nodes:
            for j in range(len(item_list)):
                 if tour[:,i] == df.loc[j].at[' ASSIGNED NODE NUMBER'].item()-1:
                        v_c=1-df.loc[j].at[' WEIGHT'].item()*ratio
                        items[z]= df.loc[j].at[' PROFIT'].item()-R*Euc_matrix[tour[0,i],tour[0,i+1]]/v_c
                        z=z+1
            i=i+1
        if i==num_nodes-1:
             for j in range(len(item_list)):
                    if tour[:,i] == df.loc[j].at[' ASSIGNED NODE NUMBER'].item()-1:
                        v_c=1-df.loc[j].at[' WEIGHT'].item()*ratio
                        items[z]= df.loc[j].at[' PROFIT'].item()-R*Euc_matrix[tour[0,num_nodes-1],tour[0,0]]/v_c
                        z=z+1
    return items

def getItem_ratio(tour, df, num_nodes, item_list, R, W, tspdata, max_cap):
    Euc_matrix = distance(tspdata, num_nodes)
    items=np.zeros(len(item_list))
    k=len(item_list)
    ratio=0.9/W
    z=0
    i=1
    w_c=1
    while z<len(item_list):
        
        plan = np.array([item_list[z]])
        obj_func, tes,totale_profit = tourLength(tour,num_nodes, plan, df, k, R, max_cap, tspdata)
        items[z]=obj_func
        z=z+1                 

    return items

def getlight_Item(tour, df, num_nodes, item_list, R, W, tspdata):
    Euc_matrix = distance(tspdata, num_nodes)
    items=np.zeros(len(item_list))
    ratio=0.9/W
    z=0
    i=1
    
   
    while z<len(item_list):
        if  i+1<num_nodes:
            for j in range(len(item_list)):
                 if tour[:,i] == df.loc[j].at[' ASSIGNED NODE NUMBER'].item()-1:
                        w_c=df.loc[j].at[' WEIGHT'].item()
                        tour_lenght=0
                        tour_fin=0
                        d=0
                        f=i
                        
                        while d+1 < i+1:
                            tour_lenght+=Euc_matrix[tour[0,d],tour[0,d+1]]
                            d=d+1
                        while f+1<num_nodes:
                            tour_fin+=Euc_matrix[tour[0,f],tour[0,f+1]]
                            f=f+1
                            if f==num_nodes-1:
                                tour_fin+=Euc_matrix[tour[0,f],tour[0,0]]
                        items[z]= df.loc[j].at[' PROFIT'].item()*tour_lenght/w_c*tour_fin*R
                        z=z+1
            i=i+1
        if i==num_nodes-1:
             for j in range(len(item_list)):
                    if tour[:,i] == df.loc[j].at[' ASSIGNED NODE NUMBER'].item()-1:
                        w_c=df.loc[j].at[' WEIGHT'].item()
                        tour_lenght=0
                        tour_fin=0
                        d=0
                        f=i
                        while d+1 < i+1:
                            tour_lenght+=Euc_matrix[tour[0,d],tour[0,d+1]]
                            d=d+1
                        tour_fin=Euc_matrix[tour[0,f],tour[0,0]]
                        items[z]= df.loc[j].at[' PROFIT'].item()*tour_lenght/w_c*tour_fin*R
                        z=z+1
    return items
    
     