import numpy as np
import vimpy
from sobol_CPI import Sobol_CPI
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from loco import LOCO
import time
import os
from utils import GenToysDataset


import argparse
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convergence rates super_learner")
    parser.add_argument("--seeds", type=int, nargs="+", help="List of seeds")
    return parser.parse_args()


def main(args):
    

    for s in args.seeds:
        p = 50
        y_method='poly'

        ns = [200, 500, 1000, 5000, 10000, 20000, 30000]
        sparsity = 0.25
        cor=0.6
        cor_meth='toep'
        snr=2


        n_cal=10
        n_cal2 = 100
        n_jobs=20

        best_model=None
        dict_model=None

        rng = np.random.RandomState(s)

        importance_score=np.zeros((21, len(ns), p))
        true_importance=np.zeros(( len(ns), p))
        p_val=np.zeros((21, len(ns), p))
        executation_time = np.zeros((21, len(ns)))
        # 0 LOCO-W, 1-6 Sobol-CPI(1) (-, sqrt, n, bootstrap, n2), 7-11 Sobol-CPI(10)(-, sqrt, n, bootstrap, n2), 12-16 Sobol-CPI(100) (-, sqrt, n, bootstrap, n2), 17-21 LOCO (-, sqrt, n, bootstrap, n2)

        print("Experiment: "+str(s))
        for (i,n) in enumerate(ns):
            print("With N="+str(n))
            X, y, true_imp = GenToysDataset(n=n, d=p, cor=cor_meth, y_method=y_method, rho_toep=cor, sparsity=sparsity, seed=s, snr=snr)
            true_importance[ i]=true_imp
            start_time = time.time()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=s)
            if y_method=='hidimstats':
                model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=s)
            elif y_method =='poly':
                ntrees = np.arange(100, 300, 100)
                lr = np.arange(.01, .1, .05)
                param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
                model = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3), param_grid = param_grid, cv = 3, n_jobs=n_jobs)
            model.fit(X_train, y_train)
            tr_time = time.time()-start_time
            start_time = time.time()
            sobol_CPI= Sobol_CPI(
                estimator=model,
                imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=s),
                n_permutations=1,
                random_state=s,
                n_jobs=n_jobs)
            sobol_CPI.fit(X_train, y_train)
            imp_time = time.time()-start_time

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='emp_var')
            executation_time[1,  i] = time.time() - start_time + tr_time + imp_time
            importance_score[1,i]= cpi_importance["importance"].reshape((p,))
            p_val[1,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='corrected_sqrt')
            executation_time[2,  i] = time.time() - start_time + tr_time + imp_time
            importance_score[2,i]= cpi_importance["importance"].reshape((p,))
            p_val[2,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='corrected_n')
            executation_time[3, i] = time.time() - start_time + tr_time + imp_time
            importance_score[3,i]= cpi_importance["importance"].reshape((p,))
            p_val[3,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='corrected_n', bootstrap=True)
            executation_time[4, i] = time.time() - start_time + tr_time + imp_time
            importance_score[4,i]= cpi_importance["importance"].reshape((p,))
            p_val[4,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='corrected_sqd', bootstrap=True)
            executation_time[5, i] = time.time() - start_time + tr_time + imp_time
            importance_score[5,i]= cpi_importance["importance"].reshape((p,))
            p_val[5,i]= cpi_importance["pval"].reshape((p,))

            # Sobol-CPI(n_cal)
            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal,  p_val='emp_var')
            executation_time[6,  i] = time.time() - start_time + tr_time + imp_time
            importance_score[6,i]= cpi_importance["importance"].reshape((p,))
            p_val[6,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal,  p_val='corrected_sqrt')
            executation_time[7, i] = time.time() - start_time + tr_time + imp_time
            importance_score[7,i]= cpi_importance["importance"].reshape((p,))
            p_val[7,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal,  p_val='corrected_n')
            executation_time[8,  i] = time.time() - start_time + tr_time + imp_time
            importance_score[8,i]= cpi_importance["importance"].reshape((p,))
            p_val[8,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal,  p_val='corrected_n', bootstrap=True)
            executation_time[9,  i] = time.time() - start_time + tr_time + imp_time
            importance_score[9,i]= cpi_importance["importance"].reshape((p,))
            p_val[9,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal,  p_val='corrected_sqd', bootstrap=True)
            executation_time[10, i] = time.time() - start_time + tr_time + imp_time
            importance_score[10,i]= cpi_importance["importance"].reshape((p,))
            p_val[10,i]= cpi_importance["pval"].reshape((p,))

            #Sobol-CPI(ncal2)
            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal2,  p_val='emp_var')
            executation_time[11, i] = time.time() - start_time + tr_time + imp_time
            importance_score[11,i]= cpi_importance["importance"].reshape((p,))
            p_val[11,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal2,  p_val='corrected_sqrt')
            executation_time[12, i] = time.time() - start_time + tr_time + imp_time
            importance_score[12,i]= cpi_importance["importance"].reshape((p,))
            p_val[12,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal2,  p_val='corrected_n')
            executation_time[13, i] = time.time() - start_time + tr_time + imp_time
            importance_score[13,i]= cpi_importance["importance"].reshape((p,))
            p_val[13,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal2,  p_val='corrected_n', bootstrap=True)
            executation_time[14, i] = time.time() - start_time + tr_time + imp_time
            importance_score[14,i]= cpi_importance["importance"].reshape((p,))
            p_val[14,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal2,  p_val='corrected_sqd', bootstrap=True)
            executation_time[15, i] = time.time() - start_time + tr_time + imp_time
            importance_score[15,i]= cpi_importance["importance"].reshape((p,))
            p_val[15,i]= cpi_importance["pval"].reshape((p,))

            #LOCO
            start_time = time.time()
            loco = LOCO(
                estimator=model,
                random_state=s,
                loss=mean_squared_error, 
                n_jobs=n_jobs,
            )
            loco.fit(X_train, y_train)
            tr_loco_time = time.time()-start_time

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='emp_var')
            executation_time[16, i] = time.time() - start_time + tr_time + tr_loco_time
            importance_score[16,i]= loco_importance["importance"].reshape((p,))
            p_val[16,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='corrected_sqrt')
            executation_time[17, i] = time.time() - start_time + tr_time + tr_loco_time
            importance_score[17,i]= loco_importance["importance"].reshape((p,))
            p_val[17,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='corrected_n')
            executation_time[18, i] = time.time() - start_time + tr_time + tr_loco_time
            importance_score[18,i]= loco_importance["importance"].reshape((p,))
            p_val[18,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='corrected_n', bootstrap=True)
            executation_time[19, i] = time.time() - start_time + tr_time + tr_loco_time
            importance_score[19,i]= loco_importance["importance"].reshape((p,))
            p_val[19,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='corrected_sqd', bootstrap=True)
            executation_time[20, i] = time.time() - start_time + tr_time + tr_loco_time
            importance_score[20,i]= loco_importance["importance"].reshape((p,))
            p_val[20,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            #LOCO Williamson
            for j in range(p):
                print("covariate: "+str(j))
                if y_method=='hidimstats':
                    model_j=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=s)
                elif y_method =='poly':
                    ntrees = np.arange(100, 300, 100)
                    lr = np.arange(.01, .1, .05)
                    param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
                    model_j = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3), param_grid = param_grid, cv = 3, n_jobs=n_jobs)
                vimp = vimpy.vim(y = y, x = X, s = j, pred_func = model_j, measure_type = "r_squared")
                vimp.get_point_est()
                vimp.get_influence_function()
                vimp.get_se()
                vimp.get_ci()
                vimp.hypothesis_test(alpha = 0.05, delta = 0)
                importance_score[0,i,j]+=vimp.vimp_*np.var(y)
                p_val[0,  i, j]=vimp.p_value_
            executation_time[0, i] = time.time() - start_time 


    #Save the results
    f_res={}
    f_res = pd.DataFrame(f_res)
    for i in range(21):
        for j in range(len(ns)):
            f_res1={}
            if i==0:
                f_res1["method"] = ["LOCO-W"]
            elif i==1:
                f_res1["method"]=["CPI"]
            elif i==2: 
                f_res1["method"]=["CPI_sqrt"]
            elif i==3:
                f_res1["method"] = ["CPI_n"]
            elif i==4:
                f_res1["method"] = ["CPI_bt"]
            elif i==5:
                f_res1["method"] = ["CPI_sqd"]
            elif i==6:
                f_res1["method"]=["S-CPI"]
            elif i==7: 
                f_res1["method"]=["S-CPI_sqrt"]
            elif i==8:
                f_res1["method"] = ["S-CPI_n"]
            elif i==9:
                f_res1["method"] = ["S-CPI_bt"]
            elif i==10:
                f_res1["method"] = ["S-CPI_sqd"]
            elif i==11:
                f_res1["method"]=["S-CPI2"]
            elif i==12: 
                f_res1["method"]=["S-CPI2_sqrt"]
            elif i==13:
                f_res1["method"] = ["S-CPI2_n"]
            elif i==14:
                f_res1["method"] = ["S-CPI2_bt"]
            elif i==15:
                f_res1["method"] = ["S-CPI2_sqd"]
            elif i==16:
                f_res1["method"]=["LOCO"]
            elif i==17: 
                f_res1["method"]=["LOCO_sqrt"]
            elif i==18:
                f_res1["method"] = ["LOCO_n"]
            elif i==19:
                f_res1["method"] = ["LOCO_bt"]
            elif i==20:
                f_res1["method"] = ["LOCO_sqd"]
            f_res1["n"]=ns[j]
            for k in range(p):
                f_res1["imp_V"+str(k)]=importance_score[i, j, k]
                f_res1["tr_V"+str(k)] =true_importance[ j, k]
                f_res1["pval"+str(k)] = p_val[i, j, k]
            f_res1['tr_time'] = executation_time[i, j]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../results/csv/inference/inference_{y_method}_p{p}_cor{cor}_seed{s}.csv"))
    f_res.to_csv(
        csv_path,
        index=False,
        ) 
    print(f_res.head())


if __name__ == "__main__":
    args = parse_args()
    main(args)