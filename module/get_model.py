import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, 
    QuadraticDiscriminantAnalysis
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

import numpy as np

from module.common import parameter_grid


def load_model(model: str, seed: int, param: dict):
    if model == 'logistic':
        clf = LogisticRegression(random_state = seed, **param)
        
    elif model == 'dt':
        clf = DecisionTreeClassifier(random_state = seed, **param)
    
    elif model == 'rf':
        clf = RandomForestClassifier(random_state = seed, **param)
    
    elif model == 'gbt':
        clf = GradientBoostingClassifier(random_state = seed, **param)
    
    elif model == 'xgb':
        clf = XGBClassifier(random_state = seed, **param)
    
    elif model == 'lgb':
        clf = LGBMClassifier(random_state = seed, **param)
    
    elif model == 'lda':
        clf = LinearDiscriminantAnalysis(**param)
    
    elif model == 'qda':
        clf = QuadraticDiscriminantAnalysis(**param)
    
    elif model == 'plsda':
        clf = PLSRegression(**param)
        
    elif model == 'mlp':
        clf = MLPClassifier(random_state = seed, **param)
    
    return clf


def load_hyperparameter(model: str):
    if model == 'logistic':
        params_dict = {
            'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                  1, 2, 3, 4, 5, 7, 9, 11, 15, 20, 25, 30, 35, 40, 50, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
    elif model == 'dt':
        params_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 1, 2, 3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3]
        }
    
    elif model == 'rf':
        params_dict = {
            'n_estimators': [3, 5, 10, 15, 20, 30, 50, 90, 95, 
                            100, 125, 130, 150],
            'criterion': ['gini'],
            'min_samples_split': [2, 4],
            'min_samples_leaf': [1, 3],
            'max_features': ['sqrt', 'log2']
        }
    
    elif model == 'gbt':
        params_dict = {
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'n_estimators': [5, 10, 50, 100, 130],
            'max_depth': [1, 2, 3, 4]
        }
    
    elif model == 'xgb':
        params_dict = {
        'min_child_weight': [1, 2, 3, 5],
            'max_depth': [3, 6, 9],
            'gamma': np.linspace(0, 3, 10),
            'objective': ['multi:softmax'],
            'booster': ['gbtree']
        }
    
    elif model == 'lgb':
        params_dict = {
            'objective': ['multiclass'],
            'num_leaves': [15, 21, 27, 31, 33],
            'max_depth': [-1, 2],
            'n_estimators': [5, 10, 50, 100, 130],
            'min_child_samples': [10, 20, 25, 30]
        }
    
    elif model == 'lda':
        params_dict1 = {
            'solver': ['lsqr', 'eigen'],
            'shrinkage': np.logspace(-3, 0, 30)
        }
        params_dict2 = {
            'solver': ['svd'],
            'tol': np.logspace(-5, -3, 20)
        }
    
    elif model == 'qda':
        params_dict = {
            'reg_param': np.append(np.array([0]), np.logspace(-5, 0, 10)),
            'tol': np.logspace(-5, -3, 10)
        }
    
    elif model == 'plsda':
        params_dict = {
            'n_components': [1, 2, 3],
            'max_iter': [300, 500, 1000],
            'tol': np.logspace(-7, -5, 10)
        }
        
    elif model == 'mlp':
        params_dict = {
            'hidden_layer_sizes': [(50), (100, 50, 10), (100, 70, 50, 30, 10)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [50, 100, 200]
        }
    
    #
    if model == 'lda':
        params = parameter_grid(params_dict1)
        params.extend(parameter_grid(params_dict2))
    else:
        params = parameter_grid(params_dict)
    
    return params