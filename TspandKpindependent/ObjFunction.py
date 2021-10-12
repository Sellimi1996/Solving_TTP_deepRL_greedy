from Knapsacksubproblem.Interdependancy import getItem_Tour
from Knapsacksubproblem.KPComponent import getPackedItem
from TspandKpindependent.distance import distance
from TspandKpindependent.ValueFunction import tourLength
from TspandKpindependent.ValueFunction import tourLengthOpt
import numpy as np

def Traveling_theif_rewards(tour, data, df,k, max_cap, num_nodes, R):
    # select all possible items depend order of tour 
    time=0, 
    totale_profit=0, 
    bestreward=0
    items_list = getItem_Tour(tour, df, num_nodes)
    packedPlan = items_list
    bestreward, time,totale_profit, touropt, knpasack  = tourLengthOpt(tour,num_nodes, packedPlan, df, k, R, max_cap,data)

    return bestreward, time, totale_profit, knpasack, touropt