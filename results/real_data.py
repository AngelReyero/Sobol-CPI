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


data = "wdbc"
model = "NN"
alpha = 0.05


csv_files = glob.glob(f"csv/realdata/{data}_{model}_seed*.csv")
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)


# Display the first few rows of the DataFrame
print(df.head())

palette = {
    'Sobol-CPI(10)': 'cyan',
    'Sobol-CPI(10)_sqrt': 'cyan',
    'Sobol-CPI(10)_n': 'cyan',
    'Sobol-CPI(10)_n2': 'cyan',
    'Sobol-CPI(1)': 'blue',
    'Sobol-CPI(1)_sqrt': 'blue',
    'Sobol-CPI(1)_n': 'blue',
    'Sobol-CPI(1)_n2': 'blue',
    'LOCO-W': 'green',
    'LOCO': 'red',
    'LOCO_n': 'red',
    'LOCO_sqrt': 'red',
    'LOCO_n2': 'red',
    'Sobol-CPI(100)': 'purple',
    'Sobol-CPI(100)_sqrt': 'purple',
    'Sobol-CPI(100)_n': 'purple',
    'Sobol-CPI(100)_n2': 'purple',
    'Sobol-CPI(10)_bt': 'cyan',
    'Sobol-CPI(1)_bt': 'blue',
    'LOCO_bt': 'red',
    'Sobol-CPI(100)_bt': 'purple',
    'PFI': 'orange',

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

    'PFI':  "o",

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

    'PFI':  (3, 5, 1, 5),

}

imp_cols = [c for c in df.columns if c.startswith("imp_V")]
p = len(imp_cols)        # number of features
null_idx = p - 1 



df["null_importance"] = df[f"imp_V{null_idx}"]

pval_cols = [c for c in df.columns if c.startswith("pval")]
pval_nonnull = pval_cols[:-1]

df["discoveries"] = (df[pval_nonnull] < alpha).sum(axis=1)
df["type_I_error"] = (df[f"pval{null_idx}"] < alpha).astype(int)


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
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# -----------------------------
# 1. Importance of the null feature
# -----------------------------
methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W', 'PFI'] 
filtered_df = df[df['method'].isin(methods_to_plot)]

sns.lineplot(data=filtered_df, x='corr', y='null_importance', hue='method', palette=palette, ax=ax[0, 0])
if data == "diabetes":
    ax[0 , 0].set_ylim(-20, 20)
ax[0, 0].set_title("Importance of artificial null feature")
ax[0, 0].tick_params(axis="x", rotation=45)
ax[0, 0].legend().remove()
ax[0, 0].set_xlabel(r'')
# -----------------------------
# 2. Discoveries across all features
# -----------------------------
df_PFI = df[df["method"] != "PFI"]
sns.lineplot(data=df_PFI, x='corr', y='discoveries', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[0, 1])
ax[0, 1].set_title("Discoveries (p-value < 0.05)")
ax[0, 1].tick_params(axis="x", rotation=45)
ax[0, 1].legend().remove()
ax[0, 1].set_xlabel(r'')

# -----------------------------
# 3. Type-I error on null feature
# -----------------------------
sns.lineplot(data=df_PFI, x='corr', y='type_I_error', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[1, 0])
ax[1, 0].axhline(alpha, ls="--", color="black", label="α = 0.05")
ax[1, 0].legend()
ax[1, 0].set_title("Type-I Error (on null feature)")
ax[1, 0].tick_params(axis="x", rotation=45)
ax[1, 0].legend().remove()
ax[1, 0].set_xlabel(r'')
# -----------------------------
# 4. Execution time
# -----------------------------
sns.lineplot(data=df, x='corr', y='tr_time', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[1, 1])
ax[1, 1].set_title("Execution time")
ax[1, 1].tick_params(axis="x", rotation=45)
ax[1, 1].legend().remove()
ax[1, 1].set_xlabel(r'')
ax[1, 1].set_yscale('log')
fig.text(0.5, -0.02, 'Correlation', ha='center', fontsize=25)

plt.tight_layout()
plt.savefig(f"figures/realdata/{data}_{model}.pdf")
