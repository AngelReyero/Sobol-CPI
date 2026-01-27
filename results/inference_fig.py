from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
import glob


"""
This script plots the power performance of the proposed methods using corrected variance.  
In a linear setting where the true Total Sobol Index can be computed, the results also
demonstrate that the Sobol-CPI method achieves better performance, even for non-null covariates.  
"""


p =100
y_method = "hidimstats"
parallel = True
seed= 0

cor=0.6
alpha = 0.05

if parallel:
    csv_files = glob.glob(f"csv/inference/inference_{y_method}_p{p}_cor{cor}*.csv")
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
else:
    df = pd.read_csv(f"csv/inference_{y_method}_p{p}_cor{cor}.csv",)


# Display the first few rows of the DataFrame
print(df.head())

palette = {
    'Sobol-CPI(10)': 'cyan',
    'Sobol-CPI(10)_sqrt': 'cyan',
    'Sobol-CPI(10)_n': 'cyan',
    'Sobol-CPI(1)': 'blue',
    'Sobol-CPI(1)_sqrt': 'blue',
    'Sobol-CPI(1)_n': 'blue',
    'LOCO-W': 'green',
    'PFI': 'orange',
    'LOCO': 'red',
    'LOCO_n': 'red',
    'LOCO_sqrt': 'red',
    'Sobol-CPI(100)': 'purple',
    'Sobol-CPI(100)_sqrt': 'purple',
    'Sobol-CPI(100)_n': 'purple',
    'Sobol-CPI(10)_bt': 'cyan',
    'Sobol-CPI(1)_bt': 'blue',
    'LOCO_bt': 'red',
    'Sobol-CPI(100)_bt': 'purple',
}



auc_scores = []
null_imp = []
non_null = []
power = []
type_I = []

def toep (d, rho=0.6):
  return np.array([[ (rho)**abs(i-j) for i in range(d)]for j in range(d)])
mat = toep(p, cor)
sobol_index = np.zeros(p)
for i in range(p):
    Sigma_without_j = np.delete(mat, i, axis=1)
    Sigma_without_jj = np.delete(Sigma_without_j, i, axis=0)
    sobol_index[i] = (
        mat[i, i]
        - Sigma_without_j[i, :]
        @ np.linalg.inv(Sigma_without_jj)
        @ Sigma_without_j[i, :].T
    )
    

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    # Extract the predictions for the current experiment (as a list)
    y_pred = row.filter(like="imp_V").values
    pval = row.filter(like="pval").values
    selected = pval<=alpha
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    auc = roc_auc_score(y, y_pred)
    auc_scores.append(auc)
    null_imp.append(np.mean(abs(y_pred[y==0])))
    non_null.append(np.mean(abs(y_pred[y==1]-sobol_index[y==1])))
    power.append(sum(selected[y==1])/sum(y==1))
    type_I.append(sum(selected[y==0])/sum(y==0))



# Add the AUC scores as a new column to the DataFrame
df['AUC'] = auc_scores
df['null_imp'] = null_imp
df['non_null'] = non_null
df['power'] = power
df['type_I'] = type_I

plt.figure()
sns.set(rc={'figure.figsize':(6,3)})


df['method'] = df['method'].replace('CPI', 'Sobol-CPI(1)')
df['method'] = df['method'].replace('S-CPI', 'Sobol-CPI(10)')
df['method'] = df['method'].replace('S-CPI2', 'Sobol-CPI(100)')
df['method'] = df['method'].replace('CPI_n', 'Sobol-CPI(1)_n')
df['method'] = df['method'].replace('S-CPI_n', 'Sobol-CPI(10)_n')
df['method'] = df['method'].replace('S-CPI2_n', 'Sobol-CPI(100)_n')
df['method'] = df['method'].replace('CPI_sqd', 'Sobol-CPI(1)_n2')
df['method'] = df['method'].replace('S-CPI_sqd', 'Sobol-CPI(10)_n2')
df['method'] = df['method'].replace('S-CPI2_sqd', 'Sobol-CPI(100)_n2')
df['method'] = df['method'].replace('CPI_sqrt', 'Sobol-CPI(1)_sqrt')
df['method'] = df['method'].replace('S-CPI_sqrt', 'Sobol-CPI(10)_sqrt')
df['method'] = df['method'].replace('S-CPI2_sqrt', 'Sobol-CPI(100)_sqrt')
df['method'] = df['method'].replace('CPI_bt', 'Sobol-CPI(1)_bt')
df['method'] = df['method'].replace('S-CPI_bt', 'Sobol-CPI(10)_bt')
df['method'] = df['method'].replace('S-CPI2_bt', 'Sobol-CPI(100)_bt')
df['method'] = df['method'].replace('LOCO_sqd', 'LOCO_n2')



sns.set_style("white")
fig, ax = plt.subplots(1, 4, figsize=(14, 3), gridspec_kw={'wspace': 0.3})

# Plot 1: AUC
methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W']
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='AUC', hue='method', palette=palette, ax=ax[0])
ax[0].set_xscale('log')
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
ax[0].set_xlabel('')
ax[0].set_title('AUC', fontsize=20)
ax[0].set_ylabel("")
ax[0].legend().remove()

# Plot 2: Bias non-null covariates
sns.lineplot(data=filtered_df, x='n', y='non_null', hue='method', palette=palette, ax=ax[1])
ax[1].set_xscale('log')
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[1].set_xlabel('')
ax[1].set_title('Bias non-null', fontsize=20)
ax[1].set_ylabel("")
ax[1].legend().remove()

# Plot 3: Power
methods_to_plot_bt = ['Sobol-CPI(1)_bt', 'Sobol-CPI(10)_bt', 'Sobol-CPI(100)_bt', 'LOCO_bt', 'LOCO-W']
filtered_df_bt = df[df['method'].isin(methods_to_plot_bt)]
sns.lineplot(data=filtered_df_bt, x='n', y='power', hue='method', palette=palette, ax=ax[2])
ax[2].set_xscale('log')
ax[2].tick_params(axis='x', labelsize=15)
ax[2].tick_params(axis='y', labelsize=15)
ax[2].set_xlabel('')
ax[2].set_title('Power', fontsize=20)
ax[2].set_ylabel("")
ax[2].legend().remove()

# Plot 4: Type-I error
sns.lineplot(data=filtered_df_bt, x='n', y='type_I', hue='method', palette=palette, ax=ax[3])
ax[3].set_xscale('log')
ax[3].tick_params(axis='x', labelsize=15)
ax[3].tick_params(axis='y', labelsize=15)
ax[3].set_xlabel('')
ns = [200, 500, 1000, 5000, 10000, 20000, 30000]
ax[3].plot(ns, [0.05 for _ in ns], linestyle='--', linewidth=1, color="black")
#ax[3].axhline(y=0.05, color='black', linestyle='--', linewidth=1)
ax[3].set_title('Type-I error', fontsize=20)
ax[3].set_ylabel("")
ax[3].legend().remove()

fig.text(0.5, -0.08, 'Number of samples', ha='center', fontsize=20)

fig.subplots_adjust(wspace=0.1)   # horizontal spacing
fig.subplots_adjust(hspace=0.01) 

plt.savefig(f"figures/inference_p{p}_cor{cor}_{y_method}.pdf", bbox_inches="tight")


