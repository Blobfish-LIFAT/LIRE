from numpy import array
from numpy import matrix
from numpy import asarray
from numpy import concatenate
from numpy import mean
from numpy import std
from numpy import float
import pandas as pd
import re

df = pd.read_csv('preliminary_result_parameters_search.csv',header=0)
list_df = df.values.tolist()
result = []
columns = ['n_dim_UMAP', 'min_dist_UMAP', 'n_neighbor_UMAP', 'robustness@5', 'robustness@10', 'robustness@15','mae', 'user_id', 'item_id']
for line in list_df:
    l = []
    l.append(line[1])
    l.append(line[2])
    l.append(line[3])
    robs = re.findall(r'\d+(?:\.\d+)?', line[4])
    for rob in robs:
        l.append(float(rob))
    l.append(line[5])
    l.append(line[6])
    l.append(line[7])
    result.append(l)
    print()
print()
df = pd.DataFrame(result,columns=columns)
result = []

min_dist_umap = [0.1,0.01,0.001]
n_comp_umap = [3,10,15]
n_neighbors_umap = [5,10,30]

columns_2 = ['min_dist','n_dim','n_neighbor','mean_mae','std_mae','mean_robustness','std_robustness']
for min_dist in min_dist_umap:
    for n_comp in n_comp_umap:
        for n_neighbors in n_neighbors_umap:
            filter1 = df["n_dim_UMAP"] == n_comp
            filter2 = df["min_dist_UMAP"] == min_dist
            filter3 = df["n_neighbor_UMAP"] == n_neighbors
            sub_space_df = df.loc[filter1 & filter2 & filter3]
            mean_mae = sub_space_df['mae'].mean()
            std_mae = sub_space_df['mae'].std()
            rob_5 = sub_space_df['robustness@5'].values
            rob_10 = sub_space_df['robustness@10'].values
            rob_15 = sub_space_df['robustness@15'].values
            rob_all_mean = mean(concatenate((rob_5, rob_10,rob_15), axis=None))
            rob_all_std = std(concatenate((rob_5, rob_10,rob_15), axis=None))
            l = [min_dist,n_comp,n_neighbors,mean_mae,std_mae,rob_all_mean,rob_all_std]
            result.append(l)

df_result = pd.DataFrame(result,columns=columns_2)
df_result.to_csv('result_mae_rob2.csv')