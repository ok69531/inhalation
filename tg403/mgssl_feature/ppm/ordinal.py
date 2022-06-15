import sys
sys.path.append('../../../')

from utils import (
      ppm_feat_load, 
      data_split,
      ParameterGrid,
      MultiCV, 
      OrdinalLogitClassifier
)

import time
import random
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
    )

# warnings.filterwarnings("ignore")

try:
      import wandb
except: 
      import sys
      import subprocess
      subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
      import wandb


wandb.login(key="1c2f31977d15e796871c32701e62c5ec1167070e")
wandb.init(project="tg403-features-ppm", entity="soyoung")
wandb.run.name = 'ordinal'
wandb.run.save()

    

def main(seed_):
    
    path = '../../data/'
    
    ppm, ppm_features, ppm_y = ppm_feat_load(path)
    train_ppm_features, train_ppm_y, test_ppm_features, test_ppm_y = data_split(
        ppm_features, 
        ppm_y.category,
        seed = seed_
    )

    
    '''
        Ordinal Regression with ppm data
    '''
    
    params_dict = {
        'random_state': [seed_], 
        'penalty': ['l1', 'l2'],
        'C': np.linspace(1e-6, 50, 150),
        'solver': ['liblinear', 'saga']
    }

    params = ParameterGrid(params_dict)

    cv_result = MultiCV(
        train_ppm_features, 
        train_ppm_y, 
        OrdinalLogitClassifier,
        params
    )

    max_tau_idx = cv_result.val_tau.argmax(axis = 0)
    best_params = cv_result.iloc[max_tau_idx][:4].to_dict()

    ordinal = OrdinalLogitClassifier(**best_params)
    ordinal.fit(train_ppm_features, train_ppm_y)
    pred = ordinal.predict(test_ppm_features)
      
    result_ = {
        'seed': seed_,
        'parameters': best_params,
        'precision': precision_score(test_ppm_y, pred, average = 'macro'), 
        'recall': recall_score(test_ppm_y, pred, average = 'macro'), 
        'f1': f1_score(test_ppm_y, pred, average = 'macro'), 
        'accuracy': accuracy_score(test_ppm_y, pred),
        'tau': stats.kendalltau(test_ppm_y, pred).correlation
      }
            

    wandb.log({
        'seed': seed_,
        'parameters': best_params,
        'precision': precision_score(test_ppm_y, pred, average = 'macro'), 
        'recall': recall_score(test_ppm_y, pred, average = 'macro'), 
        'f1': f1_score(test_ppm_y, pred, average = 'macro'), 
        'accuracy': accuracy_score(test_ppm_y, pred),
        'tau': stats.kendalltau(test_ppm_y, pred).correlation
    })
      
      
    return result_


result = []
for seed_ in range(200):
      result.append(main(seed_))
      
pd.DataFrame(result).to_csv('../../test_results/features/ppm_ordinal.csv', header = True, index = False)
wandb.finish()
