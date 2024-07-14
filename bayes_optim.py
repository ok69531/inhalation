#%%
import os
import json
import logging
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score
)
from sklearn.preprocessing import MinMaxScaler

from module.argument import get_parser
from module.read_data import (
    load_data,
    multiclass2binary,
    new_multiclass2binary
)
from module.smiles2fing import smiles2fing
from module.common import (
    data_split, 
    print_best_param
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


warnings.filterwarnings('ignore')
logging.basicConfig(format='', level=logging.INFO)


#%%
parser = get_parser()
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

args.fp_type = 'maccs'
args.cat3tohigh = True


#%%
from skopt.space import Real, Categorical, Integer

logging.info('=================================')
logging.info('tg{} {} {}'.format(args.tg_num, args.inhale_type, args.model))
logging.info('Fingerprints: {}, Use Descriptors: {}'.format(args.fp_type, args.add_md))


if (args.tg_num == 403) & (args.inhale_type == 'aerosol'):
    args.model = 
    args.fp_type = 
    args.add_md = 
    args.cat3tohigh = True
elif (args.tg_num == 403) & (args.inhale_type == 'vapour'):
    args.model = 
    args.fp_type = 
    args.add_md = 
    args.cat3tohigh = True
elif (args.tg_num == 412) & (args.inhale_type == 'aerosol'):
    # rdkit과 rdkit-md의 f1-score가 performance가 동일. auc가 더 높은 rdkit-md 사용
    args.model = 'logistic'
    args.fp_type = 'rdkit'
    args.add_md = True
    args.cat3tohigh = False
elif (args.tg_num == 412) & (args.inhale_type == 'vapour'):
    args.model = 'mlp'
    args.fp_type = 'maccs'
    args.add_md = False
    args.cat3tohigh = False
elif (args.tg_num == 413) & (args.inhale_type == 'aerosol'):
    # lda, dt, gbt 세 모델의 performance가 동일. auc가 제일 높은 dt 사용
    args.model = 'dt'
    args.fp_type = 'maccs'
    args.add_md = True
    args.cat3tohigh = False
elif (args.tg_num == 413) & (args.inhale_type == 'vapour'):
    args.model = 'rf'
    args.fp_type = 'morgan'
    args.add_md = True
    args.cat3tohigh = False


def load_model(model: str, **kwargs):
    if model == 'logistic':
        return LogisticRegression(**kwargs)
        
    elif model == 'dt':
        return DecisionTreeClassifier(**kwargs)
    
    elif model == 'rf':
        return RandomForestClassifier(**kwargs)
    
    # elif model == 'gbt':
    #     clf = GradientBoostingClassifier(random_state = seed, **param)
    
    # elif model == 'xgb':
    #     clf = XGBClassifier(random_state = seed, **param)
    
    # elif model == 'lgb':
    #     clf = LGBMClassifier(random_state = seed, **param)
    
    # elif model == 'lda':
    #     clf = LinearDiscriminantAnalysis(**param)
    
    # elif model == 'qda':
    #     clf = QuadraticDiscriminantAnalysis(**param)
    
    # elif model == 'plsda':
    #     clf = PLSRegression(**param)
        
    elif model == 'mlp':
        return MLPClassifier(**kwargs)
    
    # return clf


def get_search_space(model_name, seed):
    if model_name == 'logistic':
        params = {
            'classifier__random_state': [seed],
            'classifier__C': (0.1, 100),
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
    elif model_name == 'dt':
        params = {
            'criterion': ['gini', 'entropy'],
            'max_depth': Integer(1, 5),
            'min_samples_split': Integer(2, 5),
            'min_samples_leaf': Integer(1, 5)
        }
    elif model_name == 'rf':
        params = {
            'n_estimators': Integer(5, 150),
            'criterion': ['gini'],
            'min_samples_split': Integer(1, 4),
            'min_samples_leaf': Integer(1, 3),
            'max_features': ['sqrt', 'log2']
        }
    elif model_name == 'mlp':
        params = {
            'hidden_layer_sizes': [(50), (100, 50, 10), (100, 70, 50, 30, 10)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.001, 0.0001],
            'learning_rate_init': (0.1, 0.001),
            'max_iter': Integer(50, 200)
        }
    
    return params


x, y = load_data(path = 'data', args = args)
if args.cat3tohigh:
    y = new_multiclass2binary(y, args.tg_num)
else:
    y = multiclass2binary(y, args.tg_num)

x_train, x_test, y_train, y_test = data_split(x, y, args.splitseed)

if args.add_md:
    if args.fp_type == 'maccs':
        fp_length = 167
    elif args.fp_type == 'toxprint':
        fp_length = 729
    elif args.fp_type == 'morgan':
        fp_length = 1024
    else:
        fp_length = 2048

    train_descriptors = x_train.iloc[:, fp_length:]
    descriptors_colnames = train_descriptors.columns
    
    logging.info('Number of Descriptors: {}'.format(len(descriptors_colnames)))
    
    scaler = MinMaxScaler()
    scaled_train_descriptors = pd.DataFrame(scaler.fit_transform(train_descriptors, y_train))
    scaled_train_descriptors.columns = descriptors_colnames
    x_train.iloc[:, fp_length:] = scaled_train_descriptors

    scaled_test_descriptors = pd.DataFrame(scaler.transform(x_test.iloc[:, fp_length:]))
    scaled_test_descriptors.columns = descriptors_colnames
    x_test.iloc[:, fp_length:] = scaled_test_descriptors


#%%
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from collections import Counter
from skopt import BayesSearchCV


pipeline = ImbPipeline([
    ('smote', SMOTE(random_state = args.smoteseed, k_neighbors = args.neighbor)),
    ('classifier', load_model(args.model))
])

best_params_list = []

for seed in tqdm(range(1)):
# for seed in tqdm(range(args.num_run)):
    search_space = get_search_space(args.model, seed)
    opt = BayesSearchCV(
        estimator = pipeline,
        search_spaces = search_space,
        cv = 5,
        n_iter = 10
    )
    opt.fit(x_train, y_train)
    
    best_params_list.append(frozenset(opt.best_params_.items()))

# %%
test_prec_list = []
test_rec_list = []
test_f1_list = []
test_acc_list = []
test_auc_list = []

# smote = SMOTE(random_state = args.smoteseed, k_neighbors = args.neighbor)
# train_x, train_y = smote.fit_resample(train_x, train_y)
for i in range(len(best_params_list)):
    best_params = dict(best_params_list[i])
    best_params = {k.split('__')[-1]: v for k, v in best_params.items()}
    
    model = load_model(args.model, **best_params)
    


model = LogisticRegression(**dict(best_params_list[0]))
model.fit(x_train, y_train)
pred = model.predict(x_test)

precision_score(y_test, pred)
recall_score(y_test, pred)
f1_score(y_test, pred)
accuracy_score(y_test, pred)
# roc_auc_score(y_train, pred_score)
