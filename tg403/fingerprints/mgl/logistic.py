import sys
sys.path.append('../../../')

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
wandb.init(project="tg403-time-mgl", entity="soyoung")
wandb.run.name = 'logistic'
wandb.run.save()


def main(seed_):
      
      path = '../../data/'

      mgl, mgl_x, mgl_y = mgl_fing_load(path)
      train_x, train_y, test_x, test_y = data_split(
            mgl_x,
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
            Logistic Regression with mg/l data
      '''
      
      params_dict = {
            'random_state': [seed_], 
            'penalty': ['l1', 'l2'],
            # 'C': np.logspace(-10, 10, num = 100, base = 2),
            'C': np.linspace(1e-6, 50, 50),
            'solver': ['liblinear', 'saga']
            }

      params = ParameterGrid(params_dict)

      mgl_logit_result = MultiCV(
            train_x, 
            train_y, 
            LogisticRegression,
            params
      )

      max_tau_idx = mgl_logit_result.val_tau.argmax(axis = 0)
      best_params = mgl_logit_result.iloc[max_tau_idx][:4].to_dict()

      logit = LogisticRegression(**best_params)
      logit.fit(train_x, train_y)
      pred = logit.predict(test_x)
      
      result_ = {
            'seed': seed_,
            'parameters': best_params,
            'precision': precision_score(test_y, pred, average = 'macro'), 
            'recall': recall_score(test_y, pred, average = 'macro'), 
            'f1': f1_score(test_y, pred, average = 'macro'), 
            'accuracy': accuracy_score(test_y, pred),
            'tau': stats.kendalltau(test_y, pred).correlation
      }
            

      wandb.log({
            'seed': seed_,
            'parameters': best_params,
            'precision': precision_score(test_y, pred, average = 'macro'), 
            'recall': recall_score(test_y, pred, average = 'macro'), 
            'f1': f1_score(test_y, pred, average = 'macro'), 
            'accuracy': accuracy_score(test_y, pred),
            'tau': stats.kendalltau(test_y, pred).correlation
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
for seed_ in range(50):
      result.append(main(seed_))
      
pd.DataFrame(result).to_csv('../../test_results/time/mgl_logit.csv', header = True, index = False)
wandb.finish()