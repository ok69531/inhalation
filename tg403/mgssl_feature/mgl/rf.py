import sys
sys.path.append('../../../')

from utils import (
      mgl_feat_load,
      data_split,
      ParameterGrid,
      MultiCV
)

import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

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
wandb.init(project="tg403-features-mgl", entity="soyoung")
wandb.run.name = 'rf'
wandb.run.save()


def main(seed_):
      
      path = '../../data/'

      mgl, mgl_features, mgl_y = mgl_feat_load(path)
      train_mgl_features, train_mgl_y, test_mgl_features, test_mgl_y = data_split(
            mgl_features,
            mgl_y.category,
            seed = seed_
      )


      '''
            Random Forest with mg/l data
      '''

      params_dict = {
            'random_state': [seed_], 
            'n_estimators': np.arange(60, 155, 10),
            'min_samples_split': list(range(2, 6)),
            'max_features': ['auto', 'sqrt', 'log2'],
            'class_weight': [None, {0:1.3, 1:2, 2:5.3, 3:0.6, 4:0.8}]
      }

      params = ParameterGrid(params_dict)

      cv_result = MultiCV(
            train_mgl_features, 
            train_mgl_y, 
            RandomForestClassifier,
            params
      )

      max_tau_idx = cv_result.val_tau.argmax(axis = 0)
      best_params = cv_result.iloc[max_tau_idx][:5].to_dict()

      rf = RandomForestClassifier(**best_params)
      rf.fit(train_mgl_features, train_mgl_y)
      pred = rf.predict(test_mgl_features)
      
      result_ = {
            'seed': seed_,
            'parameters': best_params,
            'precision': precision_score(test_mgl_y, pred, average = 'macro'), 
            'recall': recall_score(test_mgl_y, pred, average = 'macro'), 
            'f1': f1_score(test_mgl_y, pred, average = 'macro'), 
            'accuracy': accuracy_score(test_mgl_y, pred),
            'tau': stats.kendalltau(test_mgl_y, pred).correlation
      }
            

      wandb.log({
            'seed': seed_,
            'parameters': best_params,
            'precision': precision_score(test_mgl_y, pred, average = 'macro'), 
            'recall': recall_score(test_mgl_y, pred, average = 'macro'), 
            'f1': f1_score(test_mgl_y, pred, average = 'macro'), 
            'accuracy': accuracy_score(test_mgl_y, pred),
            'tau': stats.kendalltau(test_mgl_y, pred).correlation
            })
      
      
      return result_


result = []
for seed_ in range(200):
      result.append(main(seed_))
      
pd.DataFrame(result).to_csv('../../test_results/features/mgl_rf.csv', header = True, index = False)
wandb.finish()