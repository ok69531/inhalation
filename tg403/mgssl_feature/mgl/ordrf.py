import sys
sys.path.append('../../../')

from utils import (
      mgl_feat_load,
      data_split,
      ParameterGrid,
      MultiCV,
      OrdinalRFClassifier
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
wandb.init(project="tg403-features-mgl", entity="soyoung")
wandb.run.name = 'ordrf'
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
            Ordinal Regression with mg/l data
      '''
      
      params_dict = {
          'random_state': [seed_], 
          'n_estimators': np.arange(60, 155, 15),
          'min_samples_split': list(range(2, 6)),
          'max_features': ['auto', 'sqrt', 'log2']
      }
      
      params = ParameterGrid(params_dict)

      cv_result = MultiCV(
            train_mgl_features, 
            train_mgl_y, 
            OrdinalRFClassifier,
            params
      )

      max_tau_idx = cv_result.val_tau.argmax(axis = 0)
      best_params = cv_result.iloc[max_tau_idx][:4].to_dict()

      ordrf = OrdinalRFClassifier(**best_params)
      ordrf.fit(train_mgl_features, train_mgl_y)
      pred = ordrf.predict(test_mgl_features)
      
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

pd.DataFrame(result).to_csv('../../test_results/features/mgl_ordrf.csv', header = True, index = False)
wandb.finish()
