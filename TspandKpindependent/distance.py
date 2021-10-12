import numpy as np
def distance(data, num_nodes):
    Euc_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(0,num_nodes):
        for j in range(0,num_nodes):
            Euc_matrix[i][j] = np.sqrt((data[i,0]-data[j,0])**2 + (data[i,1]-data[j,1])**2)
    return Euc_matrix