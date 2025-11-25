from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

sns.set_style("white")

datasets = ["wdbc", "wine-red", "wine-white", "diabetes", "california"]
models   = ["lasso", "GB", "NN", "SL", "RF"]



palette = {
    'Sobol-CPI(10)': 'cyan',

    'Sobol-CPI(1)': 'blue',

    'Sobol-CPI(100)': 'purple',


    'LOCO-W': 'green',

    'LOCO': 'red',

    'PFI': 'orange',
}




# --------------------------
# AUTOMATIC LOOP
# --------------------------

for data in datasets:
    for model in models:

        print(f"\n=== Running {data} — {model} ===")

        csv_files = glob.glob(f"csv/realdata/{data}_{model}*.csv")
        if len(csv_files) == 0:
            print(f"No CSV files found for {data}-{model}. Skipping.")
            continue

        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

        # Rename methods
        rename_map = {
            'CPI': 'Sobol-CPI(1)',
            'S-CPI': 'Sobol-CPI(10)',
            'S-CPI2': 'Sobol-CPI(100)',
        }
        df["method"] = df["method"].replace(rename_map)

        # Compute stats
        imp_cols = [c for c in df.columns if c.startswith("imp_V")]
        p = len(imp_cols)
        null_idx = p - 1
        df["null_importance"] = df[f"imp_V{null_idx}"]

        # ---------------------------
        # Create figure
        # ---------------------------
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Null importance
        methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W', 'PFI']
        filtered_df = df[df['method'].isin(methods_to_plot)]
        sns.lineplot(data=filtered_df, x='corr', y='null_importance',
                     hue='method', palette=palette, ax=ax[0])
        ax[0].set_title("Importance of artificial null feature")
        ax[0].axhline(0, ls="--", color="black")
        ax[0].legend().remove()
        ax[0].set_xlabel("")



        # 2. Execution time
        sns.lineplot(data=filtered_df, x='corr', y='tr_time',
                     hue='method', palette=palette, ax=ax[1])
        ax[1].set_title("Execution time (log scale)")
        ax[1].set_yscale("log")
        ax[1].legend().remove()
        ax[1].set_xlabel("")

        fig.text(0.5, -0.02, 'Correlation', ha='center', fontsize=25)
        plt.tight_layout()

        os.makedirs("figures/realdata", exist_ok=True)
        out_path = f"figures/realdata/pfi_{data}_{model}.pdf"
        plt.savefig(out_path)
        plt.close()


