from loco import LOCO
from hidimstat.permutation_importance import PermutationImportance
import numpy as np
import vimpy
from sobol_CPI import Sobol_CPI
from utils import GenToysDataset
import pandas as pd
from utils import best_mod
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import time
import os


y_method = 'poly'

p = 50
n = 1000
sparsity = 0.25
cor_meth='toep'
intra_cor=[0,0.15, 0.3, 0.5, 0.65, 0.85]
super_learner=False

seed= 0
num_rep=50


n_cal=100
n_jobs=10

best_model=None
dict_model=None

rng = np.random.RandomState(seed)

importance_score=np.zeros((5,num_rep, len(intra_cor), p))
true_importance=np.zeros((num_rep, len(intra_cor), p))
executation_time=np.zeros((5,num_rep, len(intra_cor)))# 5 because there is 5 methods


for l in range(num_rep):
    print("Experiment: "+str(l))
    for (i,cor) in enumerate(intra_cor):
        print("With correlation="+str(cor))
        X, y, true_imp = GenToysDataset(n=n, d=p, cor=cor_meth, y_method=y_method, rho_toep=cor, sparsity=sparsity, seed=seed)
        true_importance[l, i]=true_imp
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        start_time = time.time()
        if super_learner:
            model=best_mod(X_train, y_train, seed=seed, regressor=best_model, dict_reg=dict_model,super_learner=super_learner)
        else:
            ntrees = np.arange(100, 500, 100)
            lr = np.arange(.01, .1, .05)
            param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
            model = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3), param_grid = param_grid, cv = 5, n_jobs=n_jobs)
            model.fit(X_train, y_train)
        training_time = time.time()-start_time
        start_time = time.time()
        sobol_cpi= Sobol_CPI(
            estimator=model,
            imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5),
            n_permutations=1,
            random_state=seed,
            n_jobs=n_jobs)
        sobol_cpi.fit(X_train, y_train)
        imputation_time = time.time()-start_time
        sobol_importance = sobol_cpi.score(X_test, y_test, n_cal=n_cal)
        importance_score[4,l,i]= sobol_importance["importance"].reshape((p,))
        executation_time[4, l, i] = time.time() - start_time + training_time + imputation_time
        start_time = time.time()
        cpi_importance = sobol_cpi.score(X_test, y_test, n_cal=1)
        executation_time[0, l, i] = time.time() - start_time + training_time + imputation_time
        importance_score[0,l,i]= cpi_importance["importance"].reshape((p,))

        start_time = time.time()
        pi = PermutationImportance(
            estimator=model,
            n_permutations=1,
            random_state=seed,
            n_jobs=n_jobs,
        )
        pi.fit(X_train, y_train)
        pi_importance = pi.score(X_test, y_test)
        finish_time = time.time()
        executation_time[1, l, i] = finish_time - start_time + training_time
        importance_score[1,l,i]= pi_importance["importance"].reshape((p,))

        #LOCO Williamson
        start_time = time.time()
        ntrees = np.arange(100, 500, 100)
        lr = np.arange(.01, .1, .05)
        param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
        ## set up cv objects
        cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3), param_grid = param_grid, cv = 5, n_jobs=n_jobs)
        for j in range(p):
            print("covariate: "+str(j))
            vimp = vimpy.vim(y = y, x = X, s = j, pred_func = cv_full, measure_type = "r_squared")
            vimp.get_point_est()
            vimp.get_influence_function()
            vimp.get_se()
            vimp.get_ci()
            vimp.hypothesis_test(alpha = 0.05, delta = 0)
            importance_score[2,l,i,j]+=vimp.vimp_*np.var(y)
        executation_time[2, l, i] = time.time() - start_time 

        #LOCO Hidimstat
        start_time = time.time()
        loco = LOCO(
            estimator=model,
            random_state=seed,
            loss=mean_squared_error, 
            n_jobs=n_jobs,
        )
        loco.fit(X_train, y_train)
        loco_importance = loco.score(X_test, y_test)
        finish_time = time.time()
        executation_time[3, l, i] = finish_time - start_time + training_time
        importance_score[3,l,i]= loco_importance["importance"].reshape((p,))




#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for l in range(num_rep):
    for i in range(5):#CPI, PFI, LOCO_W, LOCO_AC, Robust-Loco
        for j in range(len(intra_cor)):
            f_res1={}
            if i==0:
                f_res1["method"] = ["0.5*CPI"]
            elif i==1:
                f_res1["method"]=["PFI"]
            elif i==2: 
                f_res1["method"]=["LOCO-W"]
            elif i==3:
                f_res1["method"]=["LOCO-HD"]
            else:
                f_res1["method"]=["S-CPI"]
            f_res1["intra_cor"]=intra_cor[j]
            for k in range(p):
                f_res1["imp_V"+str(k)]=importance_score[i,l, j, k]
                f_res1["tr_V"+str(k)] =true_importance[l, j, k]
            f_res1['training_time'] = executation_time[i, l, j]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)

if super_learner:
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../results/csv/correlation_{y_method}_p{p}_n{n}_super.csv"))
    f_res.to_csv(
        csv_path,
        index=False,
        ) 
else:
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../results/csv/correlation_{y_method}_p{p}_n{n}.csv"))
    f_res.to_csv(
        csv_path,
        index=False,
        ) 
print(f_res.head())
