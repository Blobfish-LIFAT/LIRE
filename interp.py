import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv = pd.read_csv("lasso_exp_1.csv", sep=";")

sns.lineplot("sigma", "fid_mean", data=csv)
plt.show()