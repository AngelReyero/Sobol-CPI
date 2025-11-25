from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

sns.set_style("white")

datasets = ["wdbc", "wine-red", "wine-white", "diabetes"]
models   = ["lasso", "GB", "NN", "SL", "RF"]

alpha = 0.05

# -------------------------
# COLORS / MARKERS / DASHES
# -------------------------

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

# --------------------------
# AUTOMATIC LOOP
# --------------------------

for data in datasets:
    for model in models:

        print(f"\n=== Running {data} — {model} ===")

        csv_files = glob.glob(f"csv/realdata/{data}_{model}_np_seed*.csv")
        if len(csv_files) == 0:
            print(f"No CSV files found for {data}-{model}. Skipping.")
            continue

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

        # ---------------------------
        # Create figure
        # ---------------------------
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Null importance
        methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W']#, 'PFI']
        filtered_df = df[df['method'].isin(methods_to_plot)]
        sns.lineplot(data=filtered_df, x='corr', y='null_importance',
                     hue='method', palette=palette, ax=ax[0, 0])
        ax[0, 0].set_title("Importance of artificial null feature")
        ax[0, 0].axhline(0, ls="--", color="black")
        ax[0, 0].legend().remove()
        ax[0, 0].set_xlabel("")

        # 2. Discoveries
        df_PFI = df[
            (df["method"] != "PFI") &
            (~df["method"].str.endswith("_n2"))
        ]
        bad_methods = (
            df_PFI.groupby("method")["type_I_error"]
            .max()
            .loc[lambda s: s > 0.2]
            .index
        )

        # Filter them out
        df_filt = df_PFI[~df_PFI["method"].isin(bad_methods)]

        sns.lineplot(data=df_filt, x='corr', y='discoveries',
                     hue='method', palette=palette, markers=markers,
                     dashes=dashes, style='method', ax=ax[0, 1])
        ax[0, 1].set_title("Discoveries (p < 0.05)")
        ax[0, 1].legend().remove()
        ax[0, 1].set_xlabel("")

        # 3. Type I error
        sns.lineplot(data=df_filt, x='corr', y='type_I_error',
                     hue='method', palette=palette, markers=markers,
                     dashes=dashes, style='method', ax=ax[1, 0])
        ax[1, 0].axhline(alpha, ls="--", color="black")
        ax[1, 0].set_title("Type-I Error (null feature)")
        ax[1, 0].legend().remove()
        ax[1, 0].set_xlabel("")

        # 4. Execution time
        sns.lineplot(data=df_filt, x='corr', y='tr_time',
                     hue='method', palette=palette, markers=markers,
                     dashes=dashes, style='method', ax=ax[1, 1])
        ax[1, 1].set_title("Execution time (log scale)")
        ax[1, 1].set_yscale("log")
        ax[1, 1].legend().remove()
        ax[1, 1].set_xlabel("")

        fig.text(0.5, -0.02, 'Correlation', ha='center', fontsize=25)
        plt.tight_layout()

        os.makedirs("figures/realdata", exist_ok=True)
        out_path = f"figures/realdata/np_{data}_{model}.pdf"
        plt.savefig(out_path)
        plt.close()


