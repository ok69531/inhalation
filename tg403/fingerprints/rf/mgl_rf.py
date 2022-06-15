import sys
sys.path.append('../../../../')

from utils import (
      mgl_fing_load,
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
wandb.init(project="LC50-mgl-rf", entity="soyoung")


def mgl_rf_main(seed_):
      
      path = '../../data/'

      mgl, mgl_fingerprints, mgl_y = mgl_fing_load()
      train_mgl_fingerprints, train_mgl_y, test_mgl_fingerprints, test_mgl_y = data_split(
            mgl_fingerprints,
            mgl_y.category,
            seed = seed_
      )


    # print('mg/l',
    #       '\n기초통계량:\n', mgl.value.describe(),
    #       '\n분위수: ', np.quantile(mgl.value, [0.2, 0.4, 0.6, 0.8, 1]))

    #   print('범주에 포함된 데이터의 수\n', mgl_y.category.value_counts().sort_index(),
    #         '\n비율\n', mgl_y.category.value_counts(normalize = True).sort_index())

    #   print('train 범주에 포함된 데이터의 수\n', train_mgl_y.value_counts().sort_index(),
    #         '\n비율\n', train_mgl_y.value_counts(normalize = True).sort_index())

    # print('test 범주에 포함된 데이터의 수\n', test_mgl_y.value_counts().sort_index(),
    #       '\n비율\n', test_mgl_y.value_counts(normalize = True).sort_index())


      '''
            Random Forest with mg/l data
      '''

      params_dict = {
            'random_state': [seed_], 
            'n_estimators': np.arange(30, 155, 10),
            'min_samples_split': list(range(2, 9)),
            'max_features': ['auto', 'sqrt', 'log2'],
            'class_weight': [None, {0:1.3, 1:2, 2:5.3, 3:0.6, 4:0.8}]
            }

      params = ParameterGrid(params_dict)

      mgl_rf_result = MultiCV(
            train_mgl_fingerprints, 
            train_mgl_y, 
            RandomForestClassifier,
            params
      )

      max_tau_idx = mgl_rf_result.val_tau.argmax(axis = 0)
      best_params = mgl_rf_result.iloc[max_tau_idx][:5].to_dict()

      rf = RandomForestClassifier(**best_params)
      rf.fit(train_mgl_fingerprints, train_mgl_y)
      mgl_rf_pred = rf.predict(test_mgl_fingerprints)
      
      result_ = {
            'seed': seed_,
            'parameters': best_params,
            'precision': precision_score(test_mgl_y, mgl_rf_pred, average = 'macro'), 
            'recall': recall_score(test_mgl_y, mgl_rf_pred, average = 'macro'), 
            'f1': f1_score(test_mgl_y, mgl_rf_pred, average = 'macro'), 
            'accuracy': accuracy_score(test_mgl_y, mgl_rf_pred),
            'tau': stats.kendalltau(test_mgl_y, mgl_rf_pred).correlation
      }
            

      wandb.log({
            'seed': seed_,
            'parameters': best_params,
            'precision': precision_score(test_mgl_y, mgl_rf_pred, average = 'macro'), 
            'recall': recall_score(test_mgl_y, mgl_rf_pred, average = 'macro'), 
            'f1': f1_score(test_mgl_y, mgl_rf_pred, average = 'macro'), 
            'accuracy': accuracy_score(test_mgl_y, mgl_rf_pred),
            'tau': stats.kendalltau(test_mgl_y, mgl_rf_pred).correlation
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
      result.append(mgl_rf_main(seed_))
      
pd.DataFrame(result).to_csv('../test_results/fingerprints/mgl_rf.csv', header = True, index = False)
wandb.finish()