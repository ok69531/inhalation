import sys
sys.path.append('../../../../')

from utils import (
      mgl_fing_load,
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
wandb.init(project="LC50-mgl-ordrf", entity="soyoung")


def mgl_ordrf_main(seed_):
      
      path = '../../data/'

      mgl, mgl_fingerprints, mgl_y = mgl_fing_load(path)
      train_mgl_fingerprints, train_mgl_y, test_mgl_fingerprints, test_mgl_y = data_split(
            mgl_fingerprints,
            mgl_y.category,
            seed = seed_
      )


      # print('mg/l',
      #       '\n기초통계량:\n', mgl.value.describe(),
      #       '\n분위수: ', np.quantile(mgl.value, [0.2, 0.4, 0.6, 0.8, 1]))

      # print('범주에 포함된 데이터의 수\n', mgl_y.category.value_counts().sort_index(),
      #       '\n비율\n', mgl_y.category.value_counts(normalize = True).sort_index())

      # print('train 범주에 포함된 데이터의 수\n', train_mgl_y.value_counts().sort_index(),
      #       '\n비율\n', train_mgl_y.value_counts(normalize = True).sort_index())

      # print('test 범주에 포함된 데이터의 수\n', test_mgl_y.value_counts().sort_index(),
      #       '\n비율\n', test_mgl_y.value_counts(normalize = True).sort_index())


      '''
            Ordinal Regression with mg/l data
      '''
      
      params_dict = {
          'random_state': [seed_], 
          'n_estimators': np.arange(30, 155, 10),
          'min_samples_split': list(range(2, 9)),
          'max_features': ['auto', 'sqrt', 'log2']
          }

      params = ParameterGrid(params_dict)

      mgl_ordrf_result = MultiCV(
            train_mgl_fingerprints, 
            train_mgl_y, 
            OrdinalRFClassifier,
            params
      )

      max_tau_idx = mgl_ordrf_result.val_tau.argmax(axis = 0)
      best_params = mgl_ordrf_result.iloc[max_tau_idx][:4].to_dict()

      ordrf = OrdinalRFClassifier(**best_params)
      ordrf.fit(train_mgl_fingerprints, train_mgl_y)
      mgl_ordrf_pred = ordrf.predict(test_mgl_fingerprints)
      
      result_ = {
            'seed': seed_,
            'parameters': best_params,
            'precision': precision_score(test_mgl_y, mgl_ordrf_pred, average = 'macro'), 
            'recall': recall_score(test_mgl_y, mgl_ordrf_pred, average = 'macro'), 
            'f1': f1_score(test_mgl_y, mgl_ordrf_pred, average = 'macro'), 
            'accuracy': accuracy_score(test_mgl_y, mgl_ordrf_pred),
            'tau': stats.kendalltau(test_mgl_y, mgl_ordrf_pred).correlation
      }
            

      wandb.log({
            'seed': seed_,
            'parameters': best_params,
            'precision': precision_score(test_mgl_y, mgl_ordrf_pred, average = 'macro'), 
            'recall': recall_score(test_mgl_y, mgl_ordrf_pred, average = 'macro'), 
            'f1': f1_score(test_mgl_y, mgl_ordrf_pred, average = 'macro'), 
            'accuracy': accuracy_score(test_mgl_y, mgl_ordrf_pred),
            'tau': stats.kendalltau(test_mgl_y, mgl_ordrf_pred).correlation
            })
      
      
      
      # run = neptune.init(
      # project="ok69531/LC50-mgl-logistic",
      # api_token="my_api_token",
      # ) 
      
      # run['parameters'] = best_params
      # run['precision'] = precision_score(test_mgl_y, mgl_logit_pred, average = 'macro')
      # run['recall'] = recall_score(test_mgl_y, mgl_logit_pred, average = 'macro')
      # run['f1'] = f1_score(test_mgl_y, mgl_logit_pred, average = 'macro')
      # run['accuracy'] = accuracy_score(test_mgl_y, mgl_logit_pred)
      # run['tau'] = stats.kendalltau(test_mgl_y, mgl_logit_pred).correlation
      
      # run.stop()
      
      return result_


result = []
for seed_ in range(200):
      result.append(mgl_ordrf_main(seed_))

pd.DataFrame(result).to_csv('../test_results/fingerprints/mgl_ordrf.csv', header = True, index = False)
wandb.finish()
