import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



p = 100
y_method = "hidimstats"

cor=0.6

alpha = 0.05

df = pd.read_csv(f"csv/inference_{y_method}_p{p}_cor{cor}.csv",)




print(df.head())

palette = {
    'Sobol-CPI(10)': 'purple',
    'Sobol-CPI(10)_sqrt': 'purple',
    'Sobol-CPI(10)_n': 'purple',
    'Sobol-CPI(10)_n2': 'purple',
    'Sobol-CPI(1)': 'blue',
    'Sobol-CPI(1)_sqrt': 'blue',
    'Sobol-CPI(1)_n': 'blue',
    'Sobol-CPI(1)_n2': 'blue',
    'LOCO-W': 'green',
    'LOCO': 'red',
    'LOCO_n': 'red',
    'LOCO_sqrt': 'red',
    'LOCO_n2': 'red',
    'Sobol-CPI(100)': 'cyan',
    'Sobol-CPI(100)_sqrt': 'cyan',
    'Sobol-CPI(100)_n': 'cyan',
    'Sobol-CPI(100)_n2': 'cyan',
    'Sobol-CPI(10)_bt': 'purple',
    'Sobol-CPI(1)_bt': 'blue',
    'LOCO_bt': 'red',
    'Sobol-CPI(100)_bt': 'cyan',

}

markers = {
    'Sobol-CPI(10)':  "o",
    'Sobol-CPI(10)_sqrt': "^",
    'Sobol-CPI(10)_n': "D",
    'Sobol-CPI(10)_bt': '*',
    'Sobol-CPI(10)_n2': 's',
    
    'Sobol-CPI(1)':  "o",
    'Sobol-CPI(1)_sqrt': "^",
    'Sobol-CPI(1)_n': "D",
    'Sobol-CPI(1)_bt': '*',
    'Sobol-CPI(1)_n2': 's',
    
    'Sobol-CPI(100)':  "o",
    'Sobol-CPI(100)_sqrt': "^",
    'Sobol-CPI(100)_n': "D",
    'Sobol-CPI(100)_bt': '*',
    'Sobol-CPI(100)_n2': 's',
    
    'LOCO-W':  "o",
    'LOCO':  "o",
    'LOCO_n': "D",
    'LOCO_sqrt': "^",
    'LOCO_bt': '*',
    'LOCO_n2': 's',

}


dashes = {
    'Sobol-CPI(10)':  (3, 5, 1, 5),
    'Sobol-CPI(10)_sqrt': (5, 5),
    'Sobol-CPI(10)_n': (1, 1),
    'Sobol-CPI(10)_bt': (3, 1, 3),
    'Sobol-CPI(10)_n2': (2, 4),
    
    'Sobol-CPI(1)':  (3, 5, 1, 5),
    'Sobol-CPI(1)_sqrt': (5, 5),
    'Sobol-CPI(1)_n': (1, 1),
    'Sobol-CPI(1)_bt': (3, 1, 3),
    'Sobol-CPI(1)_n2': (2, 4),
    
    'Sobol-CPI(100)':  (3, 5, 1, 5),
    'Sobol-CPI(100)_sqrt': (5, 5),
    'Sobol-CPI(100)_n': (1, 1),
    'Sobol-CPI(100)_bt': (3, 1, 3),
    'Sobol-CPI(100)_n2': (2, 4),
    
    'LOCO-W':  (3, 5, 1, 5),
    'LOCO':  (3, 5, 1, 5),
    'LOCO_n': (1, 1),
    'LOCO_sqrt': (5, 5),
    'LOCO_bt': (3, 1, 3),
    'LOCO_n2': (2, 4),

}
default_dash = (1, 1)

# Ensure that all methods have a valid dash pattern
dashes2 = {method: dashes.get(method, default_dash) for method in df['method'].unique()}


power = []
type_I = []



for index, row in df.iterrows():
    y_pred = row.filter(like="imp_V").values
    pval = row.filter(like="pval").values
    selected = pval<=alpha
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    power.append(sum(selected[y==1])/sum(y==1))
    type_I.append(sum(selected[y==0])/sum(y==0))



df['power'] = power
df['type_I'] = type_I

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})


df['method'] = df['method'].replace('CPI', 'Sobol-CPI(1)')
df['method'] = df['method'].replace('S-CPI', 'Sobol-CPI(10)')
df['method'] = df['method'].replace('S-CPI2', 'Sobol-CPI(100)')
df['method'] = df['method'].replace('CPI_n', 'Sobol-CPI(1)_n')
df['method'] = df['method'].replace('S-CPI_n', 'Sobol-CPI(10)_n')
df['method'] = df['method'].replace('S-CPI2_n', 'Sobol-CPI(100)_n')
df['method'] = df['method'].replace('CPI_sqrt', 'Sobol-CPI(1)_sqrt')
df['method'] = df['method'].replace('S-CPI_sqrt', 'Sobol-CPI(10)_sqrt')
df['method'] = df['method'].replace('S-CPI2_sqrt', 'Sobol-CPI(100)_sqrt')
df['method'] = df['method'].replace('CPI_bt', 'Sobol-CPI(1)_bt')
df['method'] = df['method'].replace('S-CPI_bt', 'Sobol-CPI(10)_bt')
df['method'] = df['method'].replace('S-CPI2_bt', 'Sobol-CPI(100)_bt')
df['method'] = df['method'].replace('CPI_sqd', 'Sobol-CPI(1)_n2')
df['method'] = df['method'].replace('S-CPI_sqd', 'Sobol-CPI(10)_n2')
df['method'] = df['method'].replace('S-CPI2_sqd', 'Sobol-CPI(100)_n2')
df['method'] = df['method'].replace('LOCO_sqd', 'LOCO_n2')




sns.set_style("white")
fig, ax = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'hspace': 0.15, 'wspace': 0.2})

methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(1)_sqrt', 'Sobol-CPI(1)_n', 'Sobol-CPI(1)_bt', 'Sobol-CPI(1)_n2'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='power', hue='method', palette=palette, markers=markers, dashes=dashes, style='method',ax=ax[0, 0])  # Top-left subplot

# Format top-left subplot
ax[0, 0].set_xscale('log')
ax[0, 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 0].tick_params(axis='y', labelsize=15) 
ax[0, 0].set_xlabel(r'')
ax[0, 0].set_ylabel(f'', fontsize=20)
#ax[0, 0].legend().remove()

methods_to_plot = ['Sobol-CPI(10)', 'Sobol-CPI(10)_sqrt', 'Sobol-CPI(10)_n', 'Sobol-CPI(10)_bt', 'Sobol-CPI(10)_n2'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='power', hue='method', palette=palette, ax=ax[0, 1], markers=markers,dashes=dashes, style='method')  # Top-right subplot

# Format top-right subplot
ax[0, 1].set_xscale('log')
ax[0, 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 1].tick_params(axis='y', labelsize=15) 
ax[0, 1].set_xlabel(r'')
ax[0, 1].set_ylabel(f'', fontsize=20)


methods_to_plot = ['Sobol-CPI(100)', 'Sobol-CPI(100)_sqrt', 'Sobol-CPI(100)_n', 'Sobol-CPI(100)_bt', 'Sobol-CPI(100)_n2'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='power', hue='method', palette=palette, ax=ax[1, 0], markers=markers, dashes=dashes, style='method')  # bottom-left subplot

ax[1, 0].set_xscale('log')
ax[1, 0].tick_params(axis='x', labelsize=15)  
ax[1, 0].tick_params(axis='y', labelsize=15) 
ax[1, 0].set_xlabel(r'')
ax[1, 0].set_ylabel(f'', fontsize=20)

methods_to_plot = ['LOCO', 'LOCO_sqrt', 'LOCO_n', 'LOCO_bt', 'LOCO_n2'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='power', hue='method', palette=palette, ax=ax[1, 1], markers=markers, dashes=dashes, style='method')  # Bottom-right subplot

ax[1, 1].set_xscale('log')
ax[1, 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1, 1].tick_params(axis='y', labelsize=15) 
ax[1, 1].set_xlabel(r'')
ax[1, 1].set_ylabel(f'', fontsize=20)

fig.text(0.5, 0.05, 'Number of samples', ha='center', fontsize=20)

fig.text(0.05, 0.45, 'Power', ha='center', fontsize=20, rotation=90)


plt.savefig(f"figures/power_{y_method}_p{p}_cor{cor}_appendix.pdf", bbox_inches="tight")



# Type-I error


sns.set_style("white")
fig, ax = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'hspace': 0.15, 'wspace': 0.2})

methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(1)_sqrt', 'Sobol-CPI(1)_n', 'Sobol-CPI(1)_bt', 'Sobol-CPI(1)_n2'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='type_I', hue='method', palette=palette, markers=markers, dashes=dashes, style='method',ax=ax[0, 0])  # Top-left subplot

ax[0, 0].set_xscale('log')
ax[0, 0].tick_params(axis='x', labelsize=15) 
ax[0, 0].tick_params(axis='y', labelsize=15) 
ax[0, 0].set_xlabel(r'')
ax[0, 0].set_ylabel(f'', fontsize=20)

methods_to_plot = ['Sobol-CPI(10)', 'Sobol-CPI(10)_sqrt', 'Sobol-CPI(10)_n', 'Sobol-CPI(10)_bt', 'Sobol-CPI(10)_n2'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='type_I', hue='method', palette=palette, ax=ax[0, 1], markers=markers,dashes=dashes, style='method')  # Top-right subplot

ax[0, 1].set_xscale('log')
ax[0, 1].tick_params(axis='x', labelsize=15) 
ax[0, 1].tick_params(axis='y', labelsize=15) 
ax[0, 1].set_xlabel(r'')
ax[0, 1].set_ylabel(f'', fontsize=20)


methods_to_plot = ['Sobol-CPI(100)', 'Sobol-CPI(100)_sqrt', 'Sobol-CPI(100)_n', 'Sobol-CPI(100)_bt', 'Sobol-CPI(100)_n2'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='type_I', hue='method', palette=palette, ax=ax[1, 0], markers=markers, dashes=dashes, style='method')  # Bottom-left subplot

ax[1, 0].set_xscale('log')
ax[1, 0].tick_params(axis='x', labelsize=15) 
ax[1, 0].tick_params(axis='y', labelsize=15) 
ax[1, 0].set_xlabel(r'')
ax[1, 0].set_ylabel(f'', fontsize=20)

methods_to_plot = ['LOCO', 'LOCO_sqrt', 'LOCO_n', 'LOCO_bt', 'LOCO_n2'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='type_I', hue='method', palette=palette, ax=ax[1, 1], markers=markers, dashes=dashes, style='method')  # Bottom-right subplot

ax[1, 1].set_xscale('log')
ax[1, 1].tick_params(axis='x', labelsize=15)  
ax[1, 1].tick_params(axis='y', labelsize=15) 
ax[1, 1].set_xlabel(r'')
ax[1, 1].set_ylabel(f'', fontsize=20)

fig.text(0.5, 0.05, 'Number of samples', ha='center', fontsize=20)

fig.text(0.05, 0.45, 'Type_I', ha='center', fontsize=20, rotation=90)


plt.savefig(f"figures/type_{y_method}_p{p}_cor{cor}_appendix.pdf", bbox_inches="tight")



