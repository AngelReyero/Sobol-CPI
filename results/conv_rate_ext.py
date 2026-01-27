import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
import glob


"""
Plot of double robustness with complex learners from the main text.
"""

parallel=True
p=50
cor=0.6
n_samples=[100, 200, 500, 1000, 2000, 5000]
super_learner=True
y_method="nonlin"

def toep (d, rho=0.6):
  return np.array([[ (rho)**abs(i-j) for i in range(d)]for j in range(d)])

def theoretical_curve(y_method, j, correlation,p, beta=[2, 1]):
    """
    Computes the theoretical value for a coordinate `j` based on the specified method.

    Parameters:
    -----------
    y_method : str
        The method used for computation. Can be either 'lin' (linear) or 'nonlin' (nonlinear).
    j : int
        The coordinate index for which the theoretical value is computed.
    correlation : float
        The correlation coefficient.
    p : int
        The dimension of the Toeplitz matrix used in the nonlinear case.
    beta : list, optional
        Coefficients used in the linear case, default is [2, 1].

    Returns:
    --------
    float
        The theoretical value for the given coordinate `j`.
    """
    if y_method == 'lin':
        return beta[j]**2*(1-correlation**2)
    elif y_method == 'nonlin':
        if j >4:
            return 0
        elif j==0:
            return (1-correlation**2)/2
        elif j==1:
            mat=toep(p, correlation)
            sigma_1=mat[1]
            sigma_1=np.delete(sigma_1, 1)
            inv=np.delete(mat, 1, axis=0)
            inv=np.delete(inv, 1, axis=1)
            inv=np.linalg.inv(inv)
            return (1-np.dot(np.dot(sigma_1,inv), sigma_1.T))*0.5
        elif j==2 or j==3:
            mat=toep(p, correlation)
            sigma_1=mat[j]
            sigma_1=np.delete(sigma_1, j)
            inv=np.delete(mat, j, axis=0)
            inv=np.delete(inv, j, axis=1)
            inv=np.linalg.inv(inv)
            return 4*(1-np.dot(np.dot(sigma_1,inv), sigma_1.T))*0.5
if parallel:
    csv_files = glob.glob(f"csv/conv_rates/conv_rates_ext_{y_method}_p{p}_cor{cor}*.csv")
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
elif super_learner:
    df = pd.read_csv(f"csv/conv_rates_ext_{y_method}_p{p}_cor{cor}_super.csv")
else:
    df = pd.read_csv(f"csv/conv_rates_{y_method}_p{p}_cor{cor}.csv")
# Display the first few rows of the DataFrame
print(df.head())

auc_scores = []
null_imp = []
# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    # Extract the predictions for the current experiment (as a list)
    y_pred = row.filter(like="imp_V").values
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    auc = roc_auc_score(y, y_pred)
    auc_scores.append(auc)
    null_imp.append(np.mean(abs(y_pred[y==0])))

# Add the AUC scores as a new column to the DataFrame
df['AUC'] = auc_scores
df['null_imp'] = null_imp


# Change method '0.5CPI' to 'S-CPI'
df['method'] = df['method'].replace('0.5*CPI', 'Sobol-CPI(1)')
df['method'] = df['method'].replace('Sobol-CPI', 'Sobol-CPI(100)')

palette = {
    'Sobol-CPI(100)': 'purple',
    'Sobol-CPI(1)': 'blue',
    'LOCO-W': 'green',
    'PFI': 'orange',
    'LOCO-HD': 'red',
    'CFI': 'brown',
    'scSAGEvf': 'salmon', 
    'PFI': 'orange'
}


df_filt = df[df['method'] != 'PFI']#Permutation feature importance only noises the figures


sns.set_style("white")

# Create figure and GridSpec
fig = plt.figure(figsize=(14, 3))
gs = gridspec.GridSpec(1, 5, width_ratios=[1, -0.15, 1, 1, 1], wspace=0.3)  # 0.1 is the spacer

# Define axes, with a spacer in position 1
ax0 = fig.add_subplot(gs[0])
ax_spacer = fig.add_subplot(gs[1])  # this will stay empty
ax1 = fig.add_subplot(gs[2])
ax2 = fig.add_subplot(gs[3])
ax3 = fig.add_subplot(gs[4])

# Turn off the spacer subplot
ax_spacer.axis("off")

# Now use [ax0, ax1, ax2, ax3] as your four real axes
axes = [ax0, ax1, ax2, ax3]
for axis in axes:
    axis.grid(False)

# Plot for imp_V0 (subplot 1)
sns.lineplot(data=df_filt, x='n_samples', y='imp_V0', hue='method', palette=palette, ax=ax0)
th_cv_v0 = theoretical_curve(y_method, 0, cor, p, beta=[2, 1])
ax0.plot(n_samples, [th_cv_v0 for _ in n_samples], label=r"Theoretical", linestyle='--', linewidth=1, color="black")
ax0.set_xscale('log')
ax0.tick_params(axis='x', labelsize=15)
ax0.tick_params(axis='y', labelsize=15)
ax0.set_xlabel('')
ax0.set_ylabel('')
ax0.set_title(f'Importance of $X_0$', fontsize=20)
ax0.legend().remove()

# Plot for imp_V6 (subplot 2)
sns.lineplot(data=df, x='n_samples', y='imp_V6', hue='method', palette=palette, ax=ax1)
th_cv_v6 = theoretical_curve(y_method, 6, cor, p, beta=[2, 1])
ax1.plot(n_samples, [th_cv_v6 for _ in n_samples], label=r"Theoretical", linestyle='--', linewidth=1, color="black")
ax1.set_xscale('log')
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_title(f'Importance of $X_6$', fontsize=20)
ax1.legend().remove()

# Plot for AUC (subplot 3)
sns.lineplot(data=df, x='n_samples', y='AUC', hue='method', palette=palette, ax=ax2)
ax2.set_xscale('log')
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_title('AUC', fontsize=20)
ax2.legend().remove()

# Plot for null importance (subplot 4)
sns.lineplot(data=df, x='n_samples', y='null_imp', hue='method', palette=palette, ax=ax3)
ax3.set_xscale('log')
ax3.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_title('Bias null', fontsize=20)
ax3.legend().remove()

# Add common x-axis label
fig.text(0.5, -0.06, 'Number of samples', ha='center', fontsize=20)

fig.subplots_adjust(wspace=0.12)   # horizontal spacing
fig.subplots_adjust(hspace=0.01) 
# Save figure
if super_learner: 
    plt.savefig(f"figures/conv_rates_ext_{y_method}_p{p}_cor{cor}_super.pdf", bbox_inches="tight")
else:
    plt.savefig(f"figures/conv_rates_ext_{y_method}_p{p}_cor{cor}.pdf", bbox_inches="tight")
