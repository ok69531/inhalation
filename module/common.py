import json
import sklearn
import numpy as np
import pandas as pd

from tqdm import tqdm

from itertools import product
from collections.abc import Iterable

from sklearn.model_selection import (
    StratifiedKFold, 
    StratifiedShuffleSplit
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score
)
from sklearn.cross_decomposition import PLSRegression


def data_split(X, y, seed):
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = seed)
    
    for train_idx, test_idx in sss.split(X, y):
        train_x = X.iloc[train_idx].reset_index(drop = True)
        train_y = y.iloc[train_idx].reset_index(drop = True)
        test_x = X.iloc[test_idx].reset_index(drop = True)
        test_y = y.iloc[test_idx].reset_index(drop = True)
    
    return train_x, test_x, train_y, test_y


def parameter_grid(param_dict):
    if not isinstance(param_dict, dict):
        raise TypeError('Parameter grid is not a dict ({!r})'.format(param_dict))
    
    if isinstance(param_dict, dict):
        for key in param_dict:
            if not isinstance(param_dict[key], Iterable):
                raise TypeError('Parameter grid value is not iterable '
                                '(key={!r}, value={!r})'.format(key, param_dict[key]))
    
    items = sorted(param_dict.items())
    keys, values = zip(*items)
    
    params_grid = []
    for v in product(*values):
        params_grid.append(dict(zip(keys, v))) 
    
    return params_grid


def cross_validation(model, x, y, seed):
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    train_precision = []
    train_recall = []
    train_f1 = []
    train_accuracy = []
    
    val_precision = []
    val_recall = []
    val_f1 = []
    val_accuracy = []
    
    for train_idx, val_idx in skf.split(x, y):
        train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
        val_x, val_y = x.iloc[val_idx], y.iloc[val_idx]
        
        
        if model == sklearn.cross_decomposition._pls.PLSRegression:
            onehot_train_y = pd.get_dummies(train_y)
            
            model.fit(train_x, onehot_train_y)
            
            train_pred = np.argmax(model.predict(train_x), axis = 1)
            val_pred = np.argmax(model.predict(val_x), axis = 1)
            
        else:
            model.fit(train_x, train_y)
            
            train_pred = model.predict(train_x)
            val_pred = model.predict(val_x)
        
        train_precision.append(precision_score(train_y, train_pred, average = 'macro'))
        train_recall.append(recall_score(train_y, train_pred, average = 'macro'))
        train_f1.append(f1_score(train_y, train_pred, average = 'macro'))
        train_accuracy.append(accuracy_score(train_y, train_pred))

        val_precision.append(precision_score(val_y, val_pred, average = 'macro'))
        val_recall.append(recall_score(val_y, val_pred, average = 'macro'))
        val_f1.append(f1_score(val_y, val_pred, average = 'macro'))
        val_accuracy.append(accuracy_score(val_y, val_pred))
        
    result = dict(zip(train_metrics + val_metrics, 
                      [np.mean(train_precision), np.mean(train_recall), np.mean(train_f1), np.mean(train_accuracy), 
                       np.mean(val_precision), np.mean(val_recall), np.mean(val_f1), np.mean(val_accuracy)]))
    
    return(result)


# def CV(x, y, model, params, seed):
#     skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    
#     metrics = ['precision', 'recall', 'f1', 'accuracy']
    
#     train_metrics = list(map(lambda x: 'train_' + x, metrics))
#     val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
#     train_precision = []
#     train_recall = []
#     train_f1 = []
#     train_accuracy = []
    
#     val_precision = []
#     val_recall = []
#     val_f1 = []
#     val_accuracy = []
    
#     for train_idx, val_idx in skf.split(x, y):
#         train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
#         val_x, val_y = x.iloc[val_idx], y.iloc[val_idx]
        
#         try:
#             clf = model(random_state = seed, **params)
#         except:
#             clf = model(**params)
        
        
#         if model == sklearn.cross_decomposition._pls.PLSRegression:
#             onehot_train_y = pd.get_dummies(train_y)
            
#             clf.fit(train_x, onehot_train_y)
            
#             train_pred = np.argmax(clf.predict(train_x), axis = 1)
#             val_pred = np.argmax(clf.predict(val_x), axis = 1)
            
#         else:
#             clf.fit(train_x, train_y)
            
#             train_pred = clf.predict(train_x)
#             val_pred = clf.predict(val_x)
        
#         train_precision.append(precision_score(train_y, train_pred, average = 'macro'))
#         train_recall.append(recall_score(train_y, train_pred, average = 'macro'))
#         train_f1.append(f1_score(train_y, train_pred, average = 'macro'))
#         train_accuracy.append(accuracy_score(train_y, train_pred))

#         val_precision.append(precision_score(val_y, val_pred, average = 'macro'))
#         val_recall.append(recall_score(val_y, val_pred, average = 'macro'))
#         val_f1.append(f1_score(val_y, val_pred, average = 'macro'))
#         val_accuracy.append(accuracy_score(val_y, val_pred))
        
#     result = dict(zip(['params'] + train_metrics + val_metrics, 
#                       [params] + [np.mean(train_precision), 
#                                   np.mean(train_recall), 
#                                   np.mean(train_f1), 
#                                   np.mean(train_accuracy), 
#                                   np.mean(val_precision), 
#                                   np.mean(val_recall), 
#                                   np.mean(val_f1), 
#                                   np.mean(val_accuracy)]))
    
#     return(result)


def metric_mean(data, metric: str):
    mean_per_hp = list(map(lambda x: np.mean(x[1]), data[metric].items()))
    return mean_per_hp


def print_best_param(val_result, metric: str):
    
    mean_list = metric_mean(val_result, metric)
    max_idx = mean_list.index(max(mean_list))
    
    best_param = val_result['model'][f'model{max_idx}']
    
    return best_param


# def load_val_result(path: str, tg_num: int, inhale_type: str, model: str):
#     try:
#         with open(f'{path}/tg{tg_num}/val_results/{inhale_type}_{model}.json', 'r') as file:
#             data = json.load(file)
#     except:
#         with open(f'{path}tg{tg_num}/val_results/{inhale_type}_{model}.json', 'r') as file:
#             data = json.load(file)
    
#     return data


# def print_best_param(path: str, tg_num: int, inhale_type: str, model: str, metric: str):
#     data = load_val_result(path, tg_num, inhale_type, model)
    
#     mean_list = metric_mean(data, 'f1')
#     max_idx = mean_list.index(max(mean_list))
    
#     best_param = data['model'][f'model{max_idx}']
    
#     return best_param