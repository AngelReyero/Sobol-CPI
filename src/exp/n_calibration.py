import numpy as np
from sobol_CPI import Sobol_CPI
from utils import GenToysDataset
import pandas as pd
from utils import best_mod
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
import os

seed= 0
num_rep=50

p=50
n=10000
intra_cor=[0,0.1, 0.3, 0.5, 0.65, 0.85]
cor_meth='toep'
y_method='nonlin'
super_learner=False


n_calib=[1, 5, 20, 50, 100, 250, 500]

n_jobs=10

best_model=None
dict_model=None

rng = np.random.RandomState(seed)

importance_score=np.zeros((len(n_calib),num_rep, len(intra_cor), p))



for l in range(num_rep):
    print("Experiment: "+str(l))
    for (i,cor) in enumerate(intra_cor):
        print("With correlation="+str(cor))
        seed+=1
        X, y, _ = GenToysDataset(n=n, d=p, cor=cor_meth, y_method=y_method, mu=None, rho_toep=cor, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
        if super_learner:
            model=best_mod(X_train, y_train, seed=seed, regressor=best_model, dict_reg=dict_model,super_learner=super_learner)
        else:
            ntrees = np.arange(100, 500, 100)
            lr = np.arange(.01, .1, .05)
            param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
            model = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3, random_state=seed), param_grid = param_grid, cv = 5, n_jobs=n_jobs)
            model.fit(X_train, y_train)
        sobol_CPI= Sobol_CPI(
            estimator=model,
            imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed),
            n_permutations=1,
            random_state=seed,
            n_jobs=n_jobs)
        sobol_CPI.fit(X_train, y_train)
        for j, n_cal in enumerate(n_calib):
            sobol_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal)
            importance_score[j,l,i]= sobol_importance["importance"].reshape((p,))
        

#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for l in range(num_rep):
    for i, n_cal in enumerate(n_calib):#n_calib
        for j in range(len(intra_cor)):
            f_res1={}
            f_res1["method"] = [f"n_cal{n_cal}"]
            f_res1["intra_cor"]=intra_cor[j]
            for k in range(p):
                f_res1["imp_V"+str(k)]=importance_score[i,l, j, k]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
if super_learner:
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../results/csv/n_cal_{y_method}_p{p}_n{n}_super.csv"))
    f_res.to_csv(
        csv_path,
        index=False,
        ) 
else:
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../results/csv/n_cal_{y_method}_p{p}_n{n}.csv"))
    f_res.to_csv(
        csv_path,
        index=False,
        ) 
print(f_res.head())
