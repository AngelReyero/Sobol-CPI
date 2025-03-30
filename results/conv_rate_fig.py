import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

"""
Plot of double robustness with complex learners from the main text.
"""


p=50
cor=0.6
n_samples=[100, 250, 500, 1000, 2000, 5000]
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

if super_learner:
    df = pd.read_csv(f"csv/conv_rates_{y_method}_p{p}_cor{cor}_super.csv")
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

df = df[df['method'] != 'PFI']#Permutation feature importance only noises the figures

# Change method '0.5CPI' to 'S-CPI'
df['method'] = df['method'].replace('0.5*CPI', 'Sobol-CPI(1)')
df['method'] = df['method'].replace('Sobol-CPI', 'Sobol-CPI(100)')

palette = {'Sobol-CPI(100)': 'purple', 'Sobol-CPI(1)': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red"}

fig, ax = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={'hspace': 0.2, 'wspace': 0.2})



# Plot for imp_V0 (top-left subplot)
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df, x='n_samples', y='imp_V0', hue='method', palette=palette, ax=ax[0, 0])  # Top-left subplot

# Add the theoretical curve for imp_V0
th_cv_v0 = theoretical_curve(y_method, 0, cor, p, beta=[2, 1])
ax[0, 0].plot(n_samples, [th_cv_v0 for _ in n_samples], label=r"Theoretical", linestyle='--', linewidth=1, color="black")

# Format top-left subplot
ax[0, 0].set_xscale('log')
ax[0, 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 0].tick_params(axis='y', labelsize=15) 
ax[0, 0].set_xlabel(r'')
ax[0, 0].set_ylabel(f'Importance of $X_0$', fontsize=20)
ax[0, 0].legend().remove()

# Plot for imp_V6 (top-right subplot)
sns.lineplot(data=df, x='n_samples', y='imp_V6', hue='method', palette=palette, ax=ax[0, 1])  # Top-right subplot

# Add the theoretical curve for imp_V6
th_cv_v6 = theoretical_curve(y_method, 6, cor, p, beta=[2, 1])
ax[0, 1].plot(n_samples, [th_cv_v6 for _ in n_samples], label=r"Theoretical", linestyle='--', linewidth=1, color="black")

# Format top-right subplot
ax[0, 1].set_xscale('log')
ax[0, 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 1].tick_params(axis='y', labelsize=15) 
ax[0, 1].set_xlabel(r'')
ax[0, 1].set_ylabel(f'Importance of $X_6$', fontsize=20)
ax[0, 1].legend().remove()

sns.lineplot(data=df, x='n_samples', y='AUC', hue='method', palette=palette, ax=ax[1, 0])  # Bottom-left subplot


# Format bottom-left subplot
ax[1, 0].set_xscale('log')
ax[1, 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1, 0].tick_params(axis='y', labelsize=15) 
ax[1, 0].set_xlabel(r'')
ax[1, 0].set_ylabel(f'AUC', fontsize=20)
ax[1, 0].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='n_samples', y='null_imp', hue='method', palette=palette, ax=ax[1, 1])  # Bottom-right subplot


# Format bottom-right subplot
ax[1, 1].set_xscale('log')
ax[1, 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1, 1].tick_params(axis='y', labelsize=15) 
ax[1, 1].set_xlabel(r'')
ax[1, 1].set_ylabel(f'Bias null covariates', fontsize=20)
ax[1, 1].legend().remove()

# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.01, right=0.99)

fig.text(0.5, 0.04, 'Number of samples', ha='center', fontsize=20)



if super_learner: 
    plt.savefig(f"figures/conv_rates_{y_method}_p{p}_cor{cor}_super.pdf", bbox_inches="tight")
else:
    plt.savefig(f"figures/conv_rates_{y_method}_p{p}_cor{cor}.pdf", bbox_inches="tight")



