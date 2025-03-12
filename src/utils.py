import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
import random
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from hidimstat.data_simulation import simu_data
from sklearn.preprocessing import PolynomialFeatures


def hypertune_predictor(estimator, X, y, param_grid, n_jobs=10):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters:
    -----------
    estimator : sklearn estimator
        The model to tune.
    X : ndarray
        Feature matrix.
    y : ndarray
        Target variable.
    param_grid : dict
        Dictionary of hyperparameters for tuning.
    n_jobs : int, optional (default=10)
        Number of jobs for parallel execution.

    Returns:
    --------
    best_estimator : sklearn estimator
        Best estimator after hyperparameter tuning.
    best_score : float
        Best score achieved during tuning.
    """
    grid_search = GridSearchCV(estimator, param_grid=param_grid, cv=2, n_jobs=n_jobs, scoring='r2')
    grid_search.fit(X, y)
    best_hyperparameters = grid_search.best_params_

    print(f"Best Hyperparameters: {best_hyperparameters}")
    return grid_search.best_estimator_, grid_search.best_score_


def best_mod(X_train, y_train, seed=2024, n_jobs=10, verbose=False, regressor=None, dict_reg=None, super_learner=False):
    """
    Find the best predictive model by hyperparameter tuning multiple regressors.

    Parameters:
    -----------
    X_train : ndarray
        Training feature matrix.
    y_train : ndarray
        Training target variable.
    seed : int, optional (default=2024)
        Random seed for reproducibility.
    n_jobs : int, optional (default=10)
        Number of parallel jobs.
    verbose : bool, optional (default=False)
        Whether to print additional information.
    regressor : sklearn estimator, optional (default=None)
        If specified, only this regressor will be tuned.
    dict_reg : dict, optional (default=None)
        Hyperparameter grid for the provided regressor.
    super_learner : bool, optional (default=False)
        Whether to use a stacked ensemble learning approach.

    Returns:
    --------
    best_model : sklearn estimator
        The best performing model.
    best_score : float (if verbose=True)
        The best score achieved.
    """
    if super_learner:
        # Define base estimators
        estimators = [
            ('rf', RandomForestRegressor(random_state=seed)),
            ('lasso', Lasso()),
            ('svr', SVR()),
            ('hgb', HistGradientBoostingRegressor(random_state=seed))
        ]

        # Define hyperparameter grid for stacking model
        param_grid = {
            'rf__n_estimators': randint(50, 500),
            'rf__max_depth': [3, 6, 10],
            'lasso__alpha': uniform(0.001, 1.0),
            'svr__C': uniform(0.1, 100),
            'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svr__gamma': ['scale', 'auto'],
            'hgb__max_iter': randint(100, 1000),
            'hgb__learning_rate': uniform(0.01, 0.3),
            'final_estimator__alpha': uniform(0.1, 10)
        }

        # Stacking Regressor with Ridge as final estimator
        stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=Ridge())

        # Hyperparameter tuning using RandomizedSearchCV
        random_search = RandomizedSearchCV(
            stacking_regressor, param_distributions=param_grid,
            n_iter=50, cv=5, random_state=seed, n_jobs=n_jobs
        )
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        best_score = random_search.best_score_

        return (best_model, best_score) if verbose else best_model

    if regressor is not None:
        model, score = hypertune_predictor(regressor, X_train, y_train, dict_reg, n_jobs=n_jobs)
        return (model, score) if verbose else model

    # List of models and their parameter grids
    models_param_grids = {
        "RandomForest": (
            RandomForestRegressor(random_state=seed),
            {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 30],
                'min_samples_split': [2, 10],
                'min_samples_leaf': [1, 4],
                'max_features': ['log2', 'sqrt'],
                'bootstrap': [True]
            }
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=seed),
            {
                'n_estimators': [100, 300],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 7],
                'min_samples_split': [2, 10],
                'min_samples_leaf': [1, 4],
                'subsample': [0.8, 1.0],
                'loss': ['squared_error', 'huber']
            }
        ),
        "XGBoost": (
            XGBRegressor(random_state=seed),
            {
                'n_estimators': [100, 300],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 7],
                'min_child_weight': [1, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1]
            }
        ),
        "Lasso": (
            Lasso(random_state=seed),
            {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'max_iter': [1000, 5000, 10000],
                'tol': [1e-4, 1e-3, 1e-2]
            }
        )
    }

    best_model, best_score = None, float('-inf')
    results = {}

    for model_name, (model, param_grid) in models_param_grids.items():
        tuned_model, score = hypertune_predictor(model, X_train, y_train, param_grid, n_jobs=n_jobs)
        results[model_name] = (tuned_model, score)
        print(f"{model_name} score: {score}")

        if score > best_score:
            best_model, best_score = tuned_model, score

    print(f"Best model: {best_model.__class__.__name__} with score {best_score}")
    
    return (best_model, best_score) if verbose else best_model


# covariance matrice 
def ind(i,j,k):
    # separates &,n into k blocks
    return int(i//k==j//k)
# One Toeplitz matrix  
def toep (d, rho=0.6):
  return np.array([[ (rho)**abs(i-j) for i in range(d)]for j in range(d)])

def GenToysDataset(n=1000, d=10, cor='toep', y_method="nonlin", k=2, mu=None, rho_toep=0.6, sparsity=0.1, seed=0, snr=2):
    """
    Generate a synthetic toy dataset for regression tasks.

    Parameters:
    -----------
    n : int, optional (default=1000)
        Number of samples.
    d : int, optional (default=10)
        Number of features.
    cor : str, optional (default='toep')
        Type of correlation among features. Options:
        - 'iso': Isotropic normal distribution.
        - 'cor': Correlated features using matrix U.
        - 'toep': Toeplitz covariance structure.
    y_method : str, optional (default='nonlin')
        Method for generating target variable y. Options:
        - 'williamson': Quadratic function of first two features.
        - 'hidimstats': High-dimensional sparse regression.
        - 'nonlin': Nonlinear interaction of first five features.
        - 'nonlin2': Extended nonlinear interactions with additional terms.
        - 'lin': Linear combination of first two features.
        - 'poly': Polynomial interactions of randomly selected features.
    k : int, optional (default=2)
        Parameter for correlation matrix U when cor='cor'.
    mu : array-like or None, optional (default=None)
        Mean vector for multivariate normal distribution.
    rho_toep : float, optional (default=0.6)
        Correlation coefficient for Toeplitz covariance matrix.
    sparsity : float, optional (default=0.1)
        Proportion of nonzero coefficients in high-dimensional regression.
    seed : int, optional (default=0)
        Random seed for reproducibility.
    snr : float, optional (default=2)
        Signal-to-noise ratio for high-dimensional regression.

    Returns:
    --------
    X : ndarray of shape (n, d)
        Feature matrix.
    y : ndarray of shape (n,)
        Target variable.
    true_imp : ndarray of shape (d,)
        Binary array indicating which features are truly important.
    """
    np.random.seed(seed)
    true_imp = np.zeros(d)
    
    if y_method == "williamson":
        X1, X2 = np.random.uniform(-1, 1, (2, n))
        X = np.column_stack((X1, X2))
        y = (25/9) * X1**2 + np.random.normal(0, 1, n)
        return X, y, np.array([1, 0])
    
    if y_method == "hidimstats":
        X, y, _, non_zero_index = simu_data(n, d, rho=rho_toep, sparsity=sparsity, seed=seed, snr=snr)
        true_imp[non_zero_index] = 1
        return X, y, true_imp
    
    mu = np.zeros(d) if mu is None else mu
    X = np.zeros((n, d))
    
    if cor == 'iso':
        X = np.random.normal(size=(n, d))
    elif cor == 'cor':
        U = np.array([[ind(i, j, k) for j in range(d)] for i in range(d)]) / np.sqrt(k)
        X = np.random.normal(size=(n, d)) @ U + mu
    elif cor == 'toep':
        X = np.random.multivariate_normal(mu, toep(d, rho_toep), size=n)
    else:
        raise ValueError("Invalid correlation type. Choose from 'iso', 'cor', or 'toep'.")
    
    if y_method == "nonlin":
        y = X[:, 0] * X[:, 1] * (X[:, 2] > 0) + 2 * X[:, 3] * X[:, 4] * (X[:, 2] <= 0)
        true_imp[:5] = 1
    elif y_method == "nonlin2":
        y = (X[:, 0] * X[:, 1] * (X[:, 2] > 0) + 2 * X[:, 3] * X[:, 4] * (X[:, 2] <= 0)
             + X[:, 5] * X[:, 6] / 2 - X[:, 7]**2 + X[:, 9] * (X[:, 8] > 0))
        true_imp[:10] = 1
    elif y_method == "lin":
        y = 2 * X[:, 0] + X[:, 1]
        true_imp[:2] = 1
    elif y_method == "poly":
        rng = np.random.RandomState(seed)
        non_zero_index = rng.choice(d, int(sparsity * d), replace=False)
        poly_transformer = PolynomialFeatures(degree=3, interaction_only=True)
        features = poly_transformer.fit_transform(X[:, non_zero_index])
        coef_features = np.random.choice([-1, 1], features.shape[1])
        y = np.dot(features, coef_features)
        true_imp[non_zero_index] = 1
    else:
        raise ValueError("Invalid y_method. Choose from 'williamson', 'hidimstats', 'nonlin', 'nonlin2', 'lin', or 'poly'.")
    
    return X, y, true_imp



def bootstrap_var(imp_list, n_groups=30, size_group=50):
    """
    Compute the variance of bootstrapped importance estimations.

    Parameters:
    -----------
    imp_list : list or array-like
        List of importance values.
    n_groups : int, optional (default=30)
        Number of bootstrap samples to generate.
    size_group : int, optional (default=50)
        Size of each bootstrap sample.

    Returns:
    --------
    float
        Variance of the estimated importance.
    """
    estim_imp = [np.mean(random.choices(imp_list, k=size_group)) for _ in range(n_groups)]
    return np.var(estim_imp)

    
