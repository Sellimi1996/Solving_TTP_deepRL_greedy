from tasks.tsp import TSPDataset, reward, render
from TspandKpindependent.ObjFunction import Traveling_theif_rewards
import datetime
import time
import numpy as np
import torch
import csv


def main(static, dynamic, x0, actor, critic, device, os, render_fn, save_dir, TSPxy, rent_ratio, weight_limit, capacity, items, num_nodes, df_loc_bag, dataName,TypKnap, file):
    # load 30 model to predict tour then compute the obj function of TTP
    def swapPositions(tab, pos1, pos2):
        vide = tab[0,pos1]
        second= tab[0,pos2]
        tab[0,pos1] = second
        tab[0,pos2] = vide
        return tab
    start  = time.time()
    t1_all = 0
    t2_all = 0
    tours=[]
    tour=[]
    rewards =[]
    obj=[]
    profit=[]
    travel_time=[]
    packedPlan=[]
    i = 0
    obj1=0

    while i < 10:
        t1 = time.time()
        ac = os.path.join(save_dir, "20/03_28_44.178385/","actor.pt")
        cri = os.path.join(save_dir, "20/03_28_44.178385/","critic.pt")
        actor.load_state_dict(torch.load(ac, device))
        critic.load_state_dict(torch.load(cri, device))
        t1_all = t1_all + time.time()-t1
         # calculate
        t2 = time.time() 
        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)
        
        tour = tour_indices.cpu().numpy()
        f=0
        for j in range(0,num_nodes):
            if tour[:,j]==0:
                f=j
        tour=swapPositions(tour, 0,f)
        tes=0
        profitmax=0
        if tour[:,0]==0:
            t2 = time.time()
            tours.append(tour_indices.cpu().numpy())
            #print('The epoch ', i,'represent the tour as follow:')
            #print( tours[i])
            # applied the objectef function of traveleing thief problem
            obj1,tes, profitmax, packitem, tour = Traveling_theif_rewards(tours[i], TSPxy, df_loc_bag, items,weight_limit,num_nodes,   rent_ratio)             # record the result
            data=[i,tes, profitmax, obj1, weight_limit]
            with open(file, 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])
                
                
            t2_all = time.time() - t2
            obj.append(obj1)
            profit.append(profitmax)
            travel_time.append(tes)
            packedPlan.append(packitem)
            name = 'Solution%d_%2.4f.png'%(i, obj[i])
            path = os.path.join('Solution', name)
            render_fn(static, tour_indices, path)
            #print('The objective function of intance is ', obj[i])
            #print('The thief picked this item ', packedPlan[i])
            #print('The profit of knapsack is :', profit[i])
            #print('The traveling time is :', travel_time[i])
            #print('Time to produce solution :', t2_all)
            i=i+1        
        


    #print("time_load_model:%2.4f"%t1_all)
    #print("time_predict_model:%2.4f"%t2_all)
    #print(time.time()-start)
    #print("The average objective function :" ,np.mean(obj))
    #plot(travel_time,profit,obj)
    return np.mean(obj), (time.time()-start), dataName ,TypKnap