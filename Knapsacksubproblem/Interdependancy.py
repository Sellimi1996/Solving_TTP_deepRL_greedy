def getItem_Tour(Tour, df, num_nodes):
    item_list=[]
    for i in range(0, num_nodes):
        for j in range(len(df)):
            if Tour[:,i] == df.loc[j].at[' ASSIGNED NODE NUMBER'].item()-1:
                item_list.append(df.loc[j].at['INDEX'].item())
    
    return item_list


def element(i_list, df):
    profit=[]
    weight=[]
    for i in range(len(i_list)):
           for j in range(len(df)):
                if i_list[i]== df.loc[j].at['INDEX'].item():
                    profit.append(df.loc[j].at[' PROFIT'].item())
                    weight.append(df.loc[j].at[' WEIGHT'].item())
    return profit,weight
