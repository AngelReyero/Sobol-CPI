import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os
import matplotlib.lines as mlines

sns.set_style("white")

datasets = ["wdbc", "wine-red", "wine-white", "diabetes", "california"]
models   = ["lasso", "GB", "NN", "SL", "RF"]

methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W', 'PFI', 'CFI', 'scSAGEvf']

palette = {
    'Sobol-CPI(10)': 'cyan',
    'Sobol-CPI(1)': 'blue',
    'Sobol-CPI(100)': 'purple',
    'LOCO-W': 'green',
    'LOCO': 'red',
    'PFI': 'orange',
    'CFI': 'brown',
    'scSAGEvf': 'salmon'
}

n_rows = len(datasets)
n_cols = len(models)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows), sharex=True, sharey=False)

for i, data in enumerate(datasets):
    for j, model in enumerate(models):

        csv_files = glob.glob(f"csv/realdata/ext_{data}_{model}*.csv")
        ax = axes[i, j]

        if len(csv_files) == 0:
            ax.set_axis_off()
            continue

        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

        rename_map = {'CPI':'Sobol-CPI(1)','S-CPI':'Sobol-CPI(10)','S-CPI2':'Sobol-CPI(100)'}
        df["method"] = df["method"].replace(rename_map)

        imp_cols = [c for c in df.columns if c.startswith("imp_V")]
        p = len(imp_cols)
        null_idx = p - 1
        df["null_importance"] = df[f"imp_V{null_idx}"]

        filtered_df = df[df['method'].isin(methods_to_plot)]

        show_legend = (i == 0 and j == 0)
        sns.lineplot(data=filtered_df, x='corr', y='null_importance',
                     hue='method', palette=palette, ax=ax, legend=False)

        ax.axhline(0, ls="--", color="black")

        if i == 0:
            ax.set_title(model, fontsize=14)
        if j == 0:
            ax.set_ylabel(data, fontsize=14)
        else:
            ax.set_ylabel("")
        if i == n_rows - 1:
            ax.set_xlabel("Correlation", fontsize=12)
        else:
            ax.set_xlabel("")

        ax.tick_params(labelsize=10)

handles = [mlines.Line2D([], [], color=palette[m], label=m)
           for m in methods_to_plot]

fig.legend(handles,
           [h.get_label() for h in handles],
           loc='lower center',
           ncol=len(handles),
           fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)


#plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig("figures/realdata/pfi_ext_all_importances.pdf")
plt.close()
