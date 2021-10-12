from Knapsacksubproblem.Interdependancy import getItem_Tour, element
from TspandKpindependent.ScoreItem import getlight_Item
from TspandKpindependent.ScoreItem import getItem_score
from TspandKpindependent.ScoreItem import getItem_ratio
from TspandKpindependent.distance import distance
from random import randrange
import numpy as np
from operator import itemgetter
from decimal import Decimal

def swapPositions(tab, pos1, pos2):
        vide = tab[0,pos1]
        second= tab[0,pos2]
        tab[0,pos1] = second
        tab[0,pos2] = vide
        return tab

def BitFlip(plan, newitem,currentitem, i):
    if newitem==currentitem:
        plan.pop(i)
    else:
        plan[i]=newitem
    return plan
def chechweight (plan, k, df):
    w_c=0
    for i in range(len(plan)):
        for j in range(0,k):
                if df.loc[j].at['INDEX'].item()==plan[i]:
                    w_c = w_c +df.loc[j].at[' WEIGHT'].item()
    return w_c
    
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

def OpmtimizeKP(tour, df, num_nodes, list_items, R, max_capacity, tspdata, p, w, k):
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
    # calculate the TTP objective function used given tour and the top score item
    bestreward, tes, totale_profit= tourLength(tour,num_nodes, item, df, k, R, max_capacity, tspdata)
    # check the other item to maximize the rewards without violate the capacity of Knapsack
    while i < len(list_items):
        index = Sorted_tab[i][0]
        if current_weight+Sorted_tab[i][2]<W:
            item.append(index)
            y=y+1
            reward,  tes, totale_profit= tourLength(tour,num_nodes, item, df, k, R, max_capacity, tspdata)
            if reward > bestreward:
                selected_item.append(index)
                bestreward=reward
                current_weight+= Sorted_tab[i][2]
            else:
                item.pop(len(item)-1)
        i=i+1
    return selected_item, bestreward

def OpmtimizeKPscore(tour, df, num_nodes, list_items, R, max_capacity, tspdata, p, w, k):
    # Compute the score of each item depend the distancs tour
    score_items = getItem_score(tour, df, num_nodes, list_items, R, max_capacity, tspdata)
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
    # calculate the TTP objective function used given tour and the top score item
    bestreward, tes, totale_profit= tourLength(tour,num_nodes, item, df, k, R, max_capacity, tspdata)
    # check the other item to maximize the rewards without violate the capacity of Knapsack
    while i < len(list_items):
        index = Sorted_tab[i][0]
        if current_weight+Sorted_tab[i][2]<W:
            item.append(index)
            y=y+1
            reward,  tes, totale_profit= tourLength(tour,num_nodes, item, df, k, R, max_capacity, tspdata)
            if reward > bestreward:
                selected_item.append(index)
                bestreward=reward
                current_weight+= Sorted_tab[i][2]
            else:
                item.pop(len(item)-1)
        i=i+1
    return selected_item, bestreward

def OpmtimizeKPbest(tour, df, num_nodes, list_items, R, max_capacity, tspdata, p, w, k):
    # Compute the score of each item depend the distancs tour
    score_items = getItem_ratio(tour, df, num_nodes, list_items, R, max_capacity, tspdata, max_capacity)
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
    # calculate the TTP objective function used given tour and the top score item
    bestreward, tes, totale_profit= tourLength(tour,num_nodes, item, df, k, R, max_capacity, tspdata)
    # check the other item to maximize the rewards without violate the capacity of Knapsack
    while i < len(list_items):
        index = Sorted_tab[i][0]
        if current_weight+Sorted_tab[i][2]<W:
            item.append(index)
            y=y+1
            reward,  tes, totale_profit= tourLength(tour,num_nodes, item, df, k, R, max_capacity, tspdata)
            if reward > bestreward:
                selected_item.append(index)
                bestreward=reward
                current_weight+= Sorted_tab[i][2]
            else:
                item.pop(len(item)-1)
        i=i+1
    return selected_item, bestreward



def tourLengthOpt(tour, num_nodes, packedPlan, df, k, R, max_cap, data):
    Euc_matrix = distance(data, num_nodes)
    w_c = 0 #current weight of knapsak
    vmax = 1 # maximum velocity
    vmin = 0.1 # minimum velocity
    tes = 0
    
    v_c = vmax-w_c*(vmax-vmin)/max_cap
    
    totale_profit_kp = 0
    d = int(num_nodes/2)
    t = 0
    
    reward = Decimal('-Infinity')
    improvement = True
    while t<1000 and improvement==True:
        improvement = False
        for i in range(0,20):
            totale_profit=0
            obj_func=0
            tes=0
            w_c=0
            v_c=vmax-w_c*(vmax-vmin)/max_cap
            f=randrange(num_nodes)
            if f!=0 and f!=d:
                 tour = swapPositions(tour, d,f)
            items_list = getItem_Tour(tour, df, num_nodes)
            p,w=element(items_list,df)
            itemopt1, rewtotal1 = OpmtimizeKP(tour, df, num_nodes, items_list, R, max_cap, data, p, w, k)
            itemopt2, rewtotal2 = OpmtimizeKPscore(tour, df, num_nodes, items_list, R, max_cap, data, p, w, k)
            itemopt3, rewtotal3 = OpmtimizeKPbest(tour, df, num_nodes, items_list, R, max_cap, data, p, w, k)
            
            if rewtotal1>rewtotal2 and rewtotal1>rewtotal3:
                    itemopt=itemopt1
            if rewtotal2>rewtotal1 and rewtotal2>rewtotal3:
                    itemopt=itemopt2
            if rewtotal3>=rewtotal1 and rewtotal3>=rewtotal2:
                    itemopt=itemopt3
             
            obj_func, tes,totale_profit = tourLength(tour,num_nodes, itemopt, df, k, R, max_cap, data)
            if reward < obj_func:
                    reward=obj_func
                    time=tes
                    profit=totale_profit
                    besttour=tour
                    knapsack=itemopt
                    improvement=True
                
            j=1
            while j<int(num_nodes/2):
                    vide=tour[0,j]
                    seconde=tour[0,num_nodes-1-j]
                    tour[0,j]=seconde
                    tour[0,num_nodes-1-j]=vide
                    j=j+1
        t=t+1
    return reward, time, profit, besttour, knapsack