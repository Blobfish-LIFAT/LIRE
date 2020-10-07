from numpy import array
from numpy import matrix
from numpy import asarray
from numpy import concatenate
from numpy import mean
from numpy import std
from numpy import float
import pandas as pd
import re

df = pd.read_csv('clustering_result_parameters_search_v2.csv',header=0)
df = df.drop(df.columns[0],axis=1)
#df = df.drop(df.columns[-1],axis=1)


hyperparameters = [None,5,10,15]
clustering_algorithms = ['kmeans','hdbscan']


columns = ['clustering_algorithm','n_cluster', 'robustness@5', 'robustness@10', 'robustness@15','mae', 'user_id', 'item_id','silhoutte_score_by_cluster','silhouette_score_all']
columns_2 = ['clustering_algorithm','n_cluster','mean_mae','std_mae','mean_robustness','std_robustness','silhouette_score','silhouette_score_by_cluster']

result = []

for hyperparameter in hyperparameters:
    for clustering_algorithm in clustering_algorithms:
        filter1 = df["clustering_algorithm"] == clustering_algorithm
        if(hyperparameter == None):
            filter2 = df["n_cluster"].isnull()
        else:
            filter2 = df["n_cluster"] == hyperparameter
        sub_space_df = df.loc[filter1 & filter2]
        mean_mae = sub_space_df['mae'].mean()
        std_mae = sub_space_df['mae'].std()
        #Moche mais une même valeur moyennée => donne cette même valeur et nous voulons cette valeur donc fuck off la propreté du code
        silhouette_score = sub_space_df['silhouette_score_all'].mean()
        if sub_space_df.empty:
            silhouette_by_cluster = None
        else:
            silhouette_by_cluster = sub_space_df['silhouette_score_by_cluster'].iloc[0]
        rob_5 = sub_space_df['robustness@5'].values
        rob_10 = sub_space_df['robustness@10'].values
        rob_15 = sub_space_df['robustness@15'].values
        rob_all_mean = mean(concatenate((rob_5, rob_10,rob_15), axis=None))
        rob_all_std = std(concatenate((rob_5, rob_10,rob_15), axis=None))
        l = [clustering_algorithm,hyperparameter,round(mean_mae,3),round(std_mae,3),round(rob_all_mean,3),round(rob_all_std,3),round(silhouette_score,3),silhouette_by_cluster]
        result.append(l)

df_result = pd.DataFrame(result,columns=columns_2)
df_result.to_csv('result_mae_rob_silh_clustering.csv')