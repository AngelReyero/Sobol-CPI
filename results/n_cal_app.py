import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

p=50
n=10000
intra_cor=[0,0.1, 0.3, 0.5, 0.65, 0.85]
y_method='nonlin'
super_learner=False

n_calib=[1, 5, 20, 50, 100, 250, 500]

def toep (d, rho=0.6):
  return np.array([[ (rho)**abs(i-j) for i in range(d)]for j in range(d)])
def theoretical_curve( j, intra_cor):
    if j >4:
        return [0 for _ in intra_cor]
    elif j==0:
        return (1-intra_cor**2)/2
    elif j==1:
        theo=[]
        for cor in intra_cor:
            mat=toep(p, cor)
            sigma_1=mat[1]
            sigma_1=np.delete(sigma_1, 1)
            inv=np.delete(mat, 1, axis=0)
            inv=np.delete(inv, 1, axis=1)
            inv=np.linalg.inv(inv)
            theo.append((1-np.dot(np.dot(sigma_1,inv), sigma_1.T))*0.5)
        return theo

if super_learner:
    df = pd.read_csv(f"csv/n_cal_{y_method}_p{p}_n{n}_super.csv",)
else:
    df = pd.read_csv(f"csv/n_cal_{y_method}_p{p}_n{n}.csv",)


# Display the first few rows of the DataFrame
print(df.head())



palette = {
    "n_cal1": "blue",
    "n_cal5": "green",
    "n_cal20": "orange",
    "n_cal50": "purple",
    "n_cal100": "red",
    "n_cal250": "brown",
    "n_cal500": "pink"
}
sns.set_style("white")
fig, ax = plt.subplots(1,2, figsize=(16, 4))


# Plot for imp_V0 (top-left subplot)
sns.set(rc={'figure.figsize':(6,3)})
sns.lineplot(data=df, x='intra_cor', y='imp_V0', hue='method', palette=palette, ax=ax[ 0]) 
th_cv= theoretical_curve(0, np.array(intra_cor))
ax[0].plot(intra_cor, th_cv, label=r"Theoretical",linestyle='--', linewidth=1, color="black")

ax[0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[ 0].set_xlabel(r'')
ax[ 0].set_ylabel(f'Importance of $X_0$', fontsize=20)
ax[ 0].legend().remove()

# Importance of V6 (not important)
sns.lineplot(data=df, x='intra_cor', y='imp_V6', hue='method', palette=palette, ax=ax[1])  # Top-right subplot
th_cv= theoretical_curve( 6, np.array(intra_cor))
ax[1].plot(intra_cor, th_cv, label=r"Theoretical",linestyle='--', linewidth=1, color="black")

ax[ 1].tick_params(axis='x', labelsize=15)  
ax[ 1].tick_params(axis='y', labelsize=15) 
ax[ 1].set_xlabel(r'')
ax[ 1].set_ylabel(f'Importance of $X_6$', fontsize=20)
ax[ 1].legend().remove()

# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.04, 'Correlation', ha='center', fontsize=20)


if super_learner:
    plt.savefig(f"figures/n_cal{y_method}_n{n}_p{p}_super.pdf", bbox_inches="tight")
else:
    plt.savefig(f"figures/n_cal{y_method}_n{n}_p{p}.pdf", bbox_inches="tight")



