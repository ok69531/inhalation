import openpyxl

import pandas as pd
import numpy as np

from tqdm import tqdm

try: 
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit-pypi"])
    # subprocess.check_call([sys.executable, "-m", "conda", "install", "rdkit", "-c conda-forge"])
    
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys


from itertools import product
from collections.abc import Iterable

import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score, 
    roc_auc_score
    )


def Smiles2Fing(smiles):
    ms_tmp = [Chem.MolFromSmiles(i) for i in smiles]
    ms_none_idx = [i for i in range(len(ms_tmp)) if ms_tmp[i] == None]
    
    ms = list(filter(None, ms_tmp))
    
    maccs = [MACCSkeys.GenMACCSKeys(i) for i in ms]
    maccs_bit = [i.ToBitString() for i in maccs]
    
    fingerprints = pd.DataFrame({'maccs': maccs_bit})
    fingerprints = fingerprints['maccs'].str.split(pat = '', n = 167, expand = True)
    fingerprints.drop(fingerprints.columns[0], axis = 1, inplace = True)
    
    colname = ['maccs_' + str(i) for i in range(1, 168)]
    fingerprints.columns = colname
    fingerprints = fingerprints.astype(int).reset_index(drop = True)
    
    return ms_none_idx, fingerprints


def mgl_fing_load(path):
    mgl = pd.read_excel(path + 'mgl.xlsx')
    
    # smiles to fingerprints
    mgl_drop_idx, mgl_fingerprints = Smiles2Fing(mgl.SMILES)
    mgl_y = mgl[['value', 'category']].drop(mgl_drop_idx).reset_index(drop = True)
    
    # quantile 기준으로 범주 구성
    # mgl_y = pd.DataFrame({'value': mgl_y, 'category': pd.qcut(mgl_y, 5, labels = range(5))})
    
    return(mgl,
           mgl_fingerprints, 
           mgl_y)


def mgl_feat_load(path):
    mgl = pd.read_excel(path + 'mgl_feature.xlsx')
    
    mgl_features = mgl[['feat' + str(i) for i in range(300)]]
    mgl_y = mgl[['value', 'category']]
    
    # quantile 기준으로 범주 구성
    # mgl_y = pd.DataFrame({'value': mgl_y, 'category': pd.qcut(mgl_y, 5, labels = range(5))})
    
    return(mgl,
           mgl_features, 
           mgl_y)


def ppm_fing_load(path):
    ppm = pd.read_excel(path + 'ppm.xlsx')

    ppm_drop_idx, ppm_fingerprints = Smiles2Fing(ppm.SMILES)
    ppm_y = ppm[['value', 'category']].drop(ppm_drop_idx).reset_index(drop = True)
    # ppm_y = pd.DataFrame({'value': ppm_y, 'category': pd.qcut(ppm_y, 5, labels = range(5))})
    
    return(ppm,
           ppm_fingerprints,
           ppm_y)


def ppm_feat_load(path):
    ppm = pd.read_excel(path + 'ppm_feature.xlsx')
    
    ppm_features = ppm[['feat' + str(i) for i in range(300)]]
    ppm_y = ppm[['value', 'category']]
    
    # quantile 기준으로 범주 구성
    # mgl_y = pd.DataFrame({'value': mgl_y, 'category': pd.qcut(mgl_y, 5, labels = range(5))})
    
    return(ppm,
           ppm_features, 
           ppm_y)


def binary_mgl_load(path):
    mgl = pd.read_excel(path + 'mgl.xlsx', sheet_name = 'Sheet1')
    
    mgl_drop_idx, mgl_fingerprints = Smiles2Fing(mgl.SMILES)
    mgl_y_ = mgl.value.drop(mgl_drop_idx).reset_index(drop = True)
    
    mgl_y = pd.DataFrame(
        {'value': mgl_y_,
         'category': pd.cut(mgl_y_, 
                            bins = [0, 0.5, np.infty], 
                            labels = [1, -1])}
        )
    
    return(mgl,
           mgl_fingerprints, 
           mgl_y)


def binary_ppm_load(path):
    ppm = pd.read_excel(path + 'ppm.xlsx', sheet_name = 'Sheet1')

    ppm_drop_idx, ppm_fingerprints = Smiles2Fing(ppm.SMILES)
    ppm_y_ = ppm.value.drop(ppm_drop_idx).reset_index(drop = True)
    
    ppm_y = pd.DataFrame(
        {'value': ppm_y_,
         'category': pd.cut(ppm_y_,
                            bins = [0, 100, np.infty], 
                            labels = [1, -1])}
        )
    
    return(ppm,
           ppm_fingerprints,
           ppm_y)


def data_split(X, y, seed):
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = seed)
    
    for train_idx, test_idx in sss.split(X, y):
        train_x = X.iloc[train_idx].reset_index(drop = True)
        train_y = y.iloc[train_idx].reset_index(drop = True)
        test_x = X.iloc[test_idx].reset_index(drop = True)
        test_y = y.iloc[test_idx].reset_index(drop = True)
    
    return train_x, train_y, test_x, test_y


def ParameterGrid(param_dict):
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


def MultiCV(x, y, model, params_grid):
    skf = StratifiedKFold(n_splits = 5)
    
    result_ = []
    metrics = ['macro_precision', 'weighted_precision', 'macro_recall', 
               'weighted_recall', 'macro_f1', 'weighted_f1', 
               'accuracy', 'tau']
    
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    for i in tqdm(range(len(params_grid))):
        train_macro_precision_, train_weighted_precision_ = [], []
        train_macro_recall_, train_weighted_recall_ = [], []
        train_macro_f1_, train_weighted_f1_ = [], []
        train_accuracy_, train_tau_ = [], []
        
        val_macro_precision_, val_weighted_precision_ = [], []
        val_macro_recall_, val_weighted_recall_ = [], []
        val_macro_f1_, val_weighted_f1_ = [], []
        val_accuracy_, val_tau_ = [], []
        
        for train_idx, val_idx in skf.split(x, y):
            train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
            val_x, val_y = x.iloc[val_idx], y.iloc[val_idx]
            
            clf = model(**params_grid[i])
            clf.fit(train_x, train_y)
            
            train_pred = clf.predict(train_x)
            val_pred = clf.predict(val_x)
            
            train_macro_precision_.append(precision_score(train_y, train_pred, average = 'macro'))
            train_weighted_precision_.append(precision_score(train_y, train_pred, average = 'weighted'))
            train_macro_recall_.append(recall_score(train_y, train_pred, average = 'macro'))
            train_weighted_recall_.append(recall_score(train_y, train_pred, average = 'weighted'))
            train_macro_f1_.append(f1_score(train_y, train_pred, average = 'macro'))
            train_weighted_f1_.append(f1_score(train_y, train_pred, average = 'weighted'))
            train_accuracy_.append(accuracy_score(train_y, train_pred))
            train_tau_.append(stats.kendalltau(train_y, train_pred))

            val_macro_precision_.append(precision_score(val_y, val_pred, average = 'macro'))
            val_weighted_precision_.append(precision_score(val_y, val_pred, average = 'weighted'))
            val_macro_recall_.append(recall_score(val_y, val_pred, average = 'macro'))
            val_weighted_recall_.append(recall_score(val_y, val_pred, average = 'weighted'))
            val_macro_f1_.append(f1_score(val_y, val_pred, average = 'macro'))
            val_weighted_f1_.append(f1_score(val_y, val_pred, average = 'weighted'))
            val_accuracy_.append(accuracy_score(val_y, val_pred))
            val_tau_.append(stats.kendalltau(val_y, val_pred))
            
        result_.append(dict(
            zip(list(params_grid[i].keys()) + train_metrics + val_metrics, 
                list(params_grid[i].values()) + 
                [np.mean(train_macro_precision_), 
                 np.mean(train_weighted_precision_),
                 np.mean(train_macro_recall_), 
                 np.mean(train_weighted_recall_),
                 np.mean(train_macro_f1_), 
                 np.mean(train_weighted_f1_),
                 np.mean(train_accuracy_), 
                 np.mean(train_tau_),
                 np.mean(val_macro_precision_), 
                 np.mean(val_weighted_precision_),
                 np.mean(val_macro_recall_), 
                 np.mean(val_weighted_recall_),
                 np.mean(val_macro_f1_), 
                 np.mean(val_weighted_f1_),
                 np.mean(val_accuracy_), 
                 np.mean(val_tau_)])))
        
    result = pd.DataFrame(result_)
    return(result)



def BinaryCV(x, y, model, params_grid):
    skf = StratifiedKFold(n_splits = 5)
    
    result_ = []
    metrics = ['macro_precision', 'macro_recall', 'macro_f1', 
               'accuracy', 'tau', 'auc']
    
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    for i in tqdm(range(len(params_grid))):
        train_macro_precision_ = []
        train_macro_recall_ = []
        train_macro_f1_ = []
        train_accuracy_  = []
        train_auc_ = []
        
        val_macro_precision_ = []
        val_macro_recall_ = []
        val_macro_f1_ = []
        val_accuracy_ = []
        val_auc_ = []
        
        for train_idx, val_idx in skf.split(x, y):
            train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
            val_x, val_y = x.iloc[val_idx], y.iloc[val_idx]
            
            clf = model(**params_grid[i])
            clf.fit(train_x, train_y)
            
            train_pred_prob = clf.predict_proba(train_x)
            val_pred_prob = clf.predict_proba(val_x)
            
            train_pred = clf.predict(train_x)
            val_pred = clf.predict(val_x)
            
            train_macro_precision_.append(precision_score(train_y, train_pred))
            train_macro_recall_.append(recall_score(train_y, train_pred))
            train_macro_f1_.append(f1_score(train_y, train_pred))
            train_accuracy_.append(accuracy_score(train_y, train_pred_prob))
            train_auc_.append(roc_auc_score(train_y, train_pred))

            val_macro_precision_.append(precision_score(val_y, val_pred))
            val_macro_recall_.append(recall_score(val_y, val_pred))
            val_macro_f1_.append(f1_score(val_y, val_pred))
            val_accuracy_.append(accuracy_score(val_y, val_pred_prob))
            val_auc_.append(roc_auc_score(val_y, val_pred))
            
        result_.append(dict(
            zip(list(params_grid[i].keys()) + train_metrics + val_metrics, 
                list(params_grid[i].values()) + 
                [np.mean(train_macro_precision_), 
                 np.mean(train_macro_recall_), 
                 np.mean(train_macro_f1_), 
                 np.mean(train_accuracy_), 
                 np.mean(train_auc_),
                 np.mean(val_macro_precision_), 
                 np.mean(val_macro_recall_), 
                 np.mean(val_macro_f1_), 
                 np.mean(val_accuracy_), 
                 np.mean(val_auc_)])))
        
    result = pd.DataFrame(result_)
    return(result)

