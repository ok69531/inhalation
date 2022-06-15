import sys
sys.path.append('../../../')

from utils import (
    ppm_feat_load,
    data_split,
    ParameterGrid,
    MultiCV,
    OrdinalRFClassifier
)

import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
    )

try:
      import wandb
except: 
      import sys
      import subprocess
      subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
      import wandb

warnings.filterwarnings("ignore")


wandb.login(key="1c2f31977d15e796871c32701e62c5ec1167070e")
wandb.init(project="tg403-features-ppm", entity="soyoung")
wandb.run.name = 'ordrf'
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
        Random Forest with ppm data
    '''

    params_dict = {
        'random_state': [seed_], 
        'n_estimators': np.arange(30, 155, 10),
        'min_samples_split': list(range(2, 9)),
        'max_features': ['auto', 'sqrt', 'log2']
    }

    params = ParameterGrid(params_dict)

    cv_result = MultiCV(
        train_ppm_features, 
        train_ppm_y, 
        OrdinalRFClassifier,
        params
    )

    max_tau_idx = cv_result.val_tau.argmax(axis = 0)
    best_params = cv_result.iloc[max_tau_idx][:4].to_dict()

    ordrf = OrdinalRFClassifier(**best_params)
    ordrf.fit(train_ppm_features, train_ppm_y)
    pred = ordrf.predict(test_ppm_features)
      
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
      
pd.DataFrame(result).to_csv('../../test_results/features/ppm_ordrf.csv', header = True, index = False)
wandb.finish()