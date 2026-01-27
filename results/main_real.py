from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from matplotlib.lines import Line2D

alpha = 0.05
palette = {
    'Sobol-CPI(10)': 'cyan',
    'Sobol-CPI(10)_sqrt': 'cyan',
    'Sobol-CPI(10)_n': 'cyan',
    'Sobol-CPI(10)_n2': 'cyan',
    'Sobol-CPI(10)_bt': 'cyan',

    'Sobol-CPI(1)': 'blue',
    'Sobol-CPI(1)_sqrt': 'blue',
    'Sobol-CPI(1)_n': 'blue',
    'Sobol-CPI(1)_n2': 'blue',
    'Sobol-CPI(1)_bt': 'blue',
    'Sobol-CPI(1)_ST': 'blue',
    'Sobol-CPI(1)_wilcox': 'blue',

    'Sobol-CPI(100)': 'purple',
    'Sobol-CPI(100)_sqrt': 'purple',
    'Sobol-CPI(100)_n': 'purple',
    'Sobol-CPI(100)_n2': 'purple',
    'Sobol-CPI(100)_bt': 'purple',

    'LOCO-W': 'green',

    'LOCO': 'red',
    'LOCO_n': 'red',
    'LOCO_sqrt': 'red',
    'LOCO_n2': 'red',
    'LOCO_bt': 'red',
    'LOCO_ST': 'red',
    'LOCO_wilcox': 'red',

    'PFI': 'orange',
}

markers = {
    'Sobol-CPI(10)': "o",
    'Sobol-CPI(10)_sqrt': "^",
    'Sobol-CPI(10)_n': "D",
    'Sobol-CPI(10)_n2': "s",
    'Sobol-CPI(10)_bt': "*",

    'Sobol-CPI(1)': "o",
    'Sobol-CPI(1)_sqrt': "^",
    'Sobol-CPI(1)_n': "D",
    'Sobol-CPI(1)_n2': "s",
    'Sobol-CPI(1)_bt': "*",
    'Sobol-CPI(1)_ST': "P",
    'Sobol-CPI(1)_wilcox': "X",

    'Sobol-CPI(100)': "o",
    'Sobol-CPI(100)_sqrt': "^",
    'Sobol-CPI(100)_n': "D",
    'Sobol-CPI(100)_n2': "s",
    'Sobol-CPI(100)_bt': "*",

    'LOCO-W': "o",
    'LOCO': "o",
    'LOCO_n': "D",
    'LOCO_sqrt': "^",
    'LOCO_n2': "s",
    'LOCO_bt': "*",
    'LOCO_ST': "P",
    'LOCO_wilcox': "X",

    'PFI': "o",
}

dashes = {
    'Sobol-CPI(10)': (3, 5, 1, 5),
    'Sobol-CPI(10)_sqrt': (5, 5),
    'Sobol-CPI(10)_n': (1, 1),
    'Sobol-CPI(10)_n2': (2, 4),
    'Sobol-CPI(10)_bt': (3, 1, 3),

    'Sobol-CPI(1)': (3, 5, 1, 5),
    'Sobol-CPI(1)_sqrt': (5, 5),
    'Sobol-CPI(1)_n': (1, 1),
    'Sobol-CPI(1)_n2': (2, 4),
    'Sobol-CPI(1)_bt': (3, 1, 3),
    'Sobol-CPI(1)_ST': (8, 3),
    'Sobol-CPI(1)_wilcox': (1, 4),

    'Sobol-CPI(100)': (3, 5, 1, 5),
    'Sobol-CPI(100)_sqrt': (5, 5),
    'Sobol-CPI(100)_n': (1, 1),
    'Sobol-CPI(100)_n2': (2, 4),
    'Sobol-CPI(100)_bt': (3, 1, 3),

    'LOCO-W': (3, 5, 1, 5),

    'LOCO': (3, 5, 1, 5),
    'LOCO_n': (1, 1),
    'LOCO_sqrt': (5, 5),
    'LOCO_n2': (2, 4),
    'LOCO_bt': (3, 1, 3),
    'LOCO_ST': (8, 3),
    'LOCO_wilcox': (1, 4),

    'PFI': (3, 5, 1, 5),
}


sns.set_style("white")

# Choose the dataset & model you want to display
MAIN_DATA = "wdbc"
MAIN_MODEL = "SL"

csv_files = glob.glob(f"csv/realdata/{MAIN_DATA}_{MAIN_MODEL}_np_seed*.csv")
if len(csv_files) == 0:
    print(f"No CSV files found for {MAIN_DATA}-{MAIN_MODEL}. Skipping.")

df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Rename methods
rename_map = {
    'CPI': 'Sobol-CPI(1)',
    'S-CPI': 'Sobol-CPI(10)',
    'S-CPI2': 'Sobol-CPI(100)',
    'CPI_n': 'Sobol-CPI(1)_n',
    'S-CPI_n': 'Sobol-CPI(10)_n',
    'S-CPI2_n': 'Sobol-CPI(100)_n',
    'CPI_sqd': 'Sobol-CPI(1)_n2',
    'S-CPI_sqd': 'Sobol-CPI(10)_n2',
    'S-CPI2_sqd': 'Sobol-CPI(100)_n2',
    'CPI_sqrt': 'Sobol-CPI(1)_sqrt',
    'S-CPI_sqrt': 'Sobol-CPI(10)_sqrt',
    'S-CPI2_sqrt': 'Sobol-CPI(100)_sqrt',
    'CPI_bt': 'Sobol-CPI(1)_bt',
    'S-CPI_bt': 'Sobol-CPI(10)_bt',
    'S-CPI2_bt': 'Sobol-CPI(100)_bt',
    'LOCO_sqd': 'LOCO_n2',
    'CPI_ST': 'Sobol-CPI(1)_ST',
    'CPI_wilcox': 'Sobol-CPI(1)_wilcox'
}
df["method"] = df["method"].replace(rename_map)

# Compute stats
imp_cols = [c for c in df.columns if c.startswith("imp_V")]
p = len(imp_cols)
null_idx = p - 1
df["null_importance"] = df[f"imp_V{null_idx}"]

pval_cols = [c for c in df.columns if c.startswith("pval")]
pval_nonnull = pval_cols[:-1]

df["discoveries"] = (df[pval_nonnull] < alpha).sum(axis=1)
df["type_I_error"] = (df[f"pval{null_idx}"] < alpha).astype(int)
# Now build MAIN figure
fig, ax = plt.subplots(1, 4, figsize=(14, 3))  # 1 row, 4 columns

# Panel 1 — Null importance
methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W']#, 'PFI']
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='corr', y='null_importance',
             hue='method', palette=palette, ax=ax[0])
ax[0].set_title("Null Importance", fontsize=20)
ax[0].set_xlabel("")
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
ax[0].set_ylabel("")
ax[0].legend().remove()



df_PFI = df[
    (df["method"] != "PFI") #&
    #(~df["method"].str.endswith("_n2"))
]
bad_methods = (
    df_PFI
    .groupby(["method", "corr"])["type_I_error"]
    .mean()                # mean per (method, correlation)
    .groupby("method")     # regroup by method
    .max()                 # max of the means across correlations
    .loc[lambda s: s > 0.2]
    .index
)

methods_inference = ['Sobol-CPI(1)_wilcox', 'Sobol-CPI(1)_bt','Sobol-CPI(10)_bt', 'Sobol-CPI(100)_bt', 'LOCO_bt', 'LOCO-W']#, 'PFI']
df_filt = df[df['method'].isin(methods_inference)]
# Filter them out
#df_filt = df_PFI[~df_PFI["method"].isin(bad_methods)]


# Panel 2 — Discoveries
sns.lineplot(data=df_filt, x='corr', y='discoveries',
             hue='method', palette=palette, markers=markers,
             dashes=dashes, style='method', ax=ax[1])
ax[1].set_title("Discoveries (p < 0.05)",fontsize=20 )
ax[1].set_xlabel("")
ax[1].set_ylabel("")
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[1].legend().remove()

# Panel 3 — Type I error
sns.lineplot(data=df_filt, x='corr', y='type_I_error',
             hue='method', palette=palette, markers=markers,
             dashes=dashes, style='method', ax=ax[2])
ax[2].set_title("Type-I Error", fontsize=20)
ax[2].axhline(alpha, ls="--", color="black")
ax[2].set_xlabel("")
ax[2].set_ylabel("")
ax[2].tick_params(axis='x', labelsize=15)
ax[2].tick_params(axis='y', labelsize=15)
ax[2].legend().remove()

# Panel 4 — Time
sns.lineplot(data=df_filt, x='corr', y='tr_time',
             hue='method', palette=palette, markers=markers,
             dashes=dashes, style='method', ax=ax[3])
ax[3].set_title("Time (s)(log-scale)", fontsize=20)
ax[3].set_yscale("log")
ax[3].set_xlabel("")
ax[3].set_ylabel("")
ax[3].tick_params(axis='x', labelsize=15)
ax[3].tick_params(axis='y', labelsize=15)
ax[3].legend().remove()

# ---- Legend (GOOD methods only) ----
good_methods = df_filt['method'].unique()

legend_handles = [
    Line2D([0], [0],
           color=palette[m],
           marker=markers.get(m, None),
           linestyle=(0, dashes[m]),
           label=m)
    for m in good_methods
]

#fig.legend(
#    handles=legend_handles,
#    loc='lower center',
#    bbox_to_anchor=(0.5, -0.08),
#    ncol=len(good_methods),
#    title="Methods"
#)
fig.text(0.5, -0.02, 'Correlation', ha='center', fontsize=25)
plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.subplots_adjust(wspace=0.16)   # horizontal spacing
fig.subplots_adjust(hspace=0.01) 
out_path = f"figures/realdata/main_{MAIN_DATA}_{MAIN_MODEL}.pdf"

# avoid cutting legend in PDF
plt.savefig(out_path, bbox_inches='tight')
