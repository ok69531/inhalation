import sys
sys.path.append('../../../')

from utils import (
      mgl_feat_load,
      data_split,
      ParameterGrid,
      MultiCV
)

import time
import random
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

import scipy.stats as stats
from sklearn.metrics import classification_report
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
wandb.run.name = 'logistic'
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
            Logistic Regression with mg/l data
      '''
      
      params_dict = {
            'random_state': [seed_], 
            'penalty': ['l1', 'l2'],
            'C': np.linspace(1e-6, 50, 30),
            'solver': ['liblinear', 'saga']
      }

      params = ParameterGrid(params_dict)

      cv_result = MultiCV(
            train_mgl_features, 
            train_mgl_y, 
            LogisticRegression,
            params
      )

      max_tau_idx = cv_result.val_tau.argmax(axis = 0)
      best_params = cv_result.iloc[max_tau_idx][:4].to_dict()

      logit = LogisticRegression(**best_params)
      logit.fit(train_mgl_features, train_mgl_y)
      pred = logit.predict(test_mgl_features)
      
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
      
pd.DataFrame(result).to_csv('../../test_results/features/mgl_logit.csv', header = True, index = False)
wandb.finish()
