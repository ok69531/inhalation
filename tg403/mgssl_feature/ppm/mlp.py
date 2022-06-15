import sys
sys.path.append('../../../')

from utils import (
    ppm_feat_load,
    data_split,
    ParameterGrid,
    model3
)

import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

import scipy.stats as stats
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
    )
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers

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
wandb.run.name = 'mlp'
wandb.run.save()



def train(x, y, seed_, lr, epochs):
    tf.random.set_seed(seed_)
    model = model3()
    
    adam = K.optimizers.Adam(lr)
    scc = K.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
    model.fit(x, y, epochs = epochs, batch_size = len(y), verbose = 0)
    
    return model
    

def MLP_CV(x, y, params_grid, seed_):
    result_ = []
    
    metrics = ['macro_precision', 'weighted_precision', 'macro_recall', 
               'weighted_recall', 'macro_f1', 'weighted_f1', 
               'accuracy', 'tau']
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    for params in tqdm(params_grid):
        
        train_macro_precision_, train_weighted_precision_ = [], []
        train_macro_recall_, train_weighted_recall_ = [], []
        train_macro_f1_, train_weighted_f1_ = [], []
        train_accuracy_, train_tau_ = [], []

        val_macro_precision_, val_weighted_precision_ = [], []
        val_macro_recall_, val_weighted_recall_ = [], []
        val_macro_f1_, val_weighted_f1_ = [], []
        val_accuracy_, val_tau_ = [], []


        skf = StratifiedKFold(n_splits = 5)

        for train_idx, test_idx in skf.split(x, y):
            train_x = x.iloc[train_idx].reset_index(drop = True)
            train_y = y.iloc[train_idx].reset_index(drop = True)
            val_x = x.iloc[test_idx].reset_index(drop = True)
            val_y = y.iloc[test_idx].reset_index(drop = True)

            train_x = tf.cast(train_x, tf.float32)
            train_y = tf.cast(train_y, tf.int32)
            val_x = tf.cast(val_x, tf.float32)
            val_y = tf.cast(val_y, tf.int32)
            
            model = train(train_x, train_y, seed_, params['learning_rate'], params['epochs'])
            
            train_pred_prob = model.predict(train_x)
            train_pred = np.argmax(train_pred_prob, axis = 1)
            
            val_pred = np.argmax(model.predict(val_x), axis = 1)
            
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
            zip(['seed'] + list(params.keys()) + train_metrics + val_metrics, 
                [seed_] + list(params.values()) + \
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


def main(seed_):
    path = '../../data/'

    ppm, ppm_features, ppm_y = ppm_feat_load(path)
    train_ppm_features, train_ppm_y, test_ppm_features, test_ppm_y = data_split(
        ppm_features, 
        ppm_y.category,
        seed = seed_
    )

    
    train_x = tf.cast(train_ppm_features, tf.float32)
    train_y = tf.cast(train_ppm_y, tf.int32)
    test_x = tf.cast(test_ppm_features, tf.float32)
    test_y = tf.cast(test_ppm_y, tf.int32)
    
    
    params_dict = {
        'learning_rate': [0.01, 0.001],
        'epochs': [1000, 10000, 25000]
    }
    params_grid = ParameterGrid(params_dict)
    
    
    cv_result = MLP_CV(
        train_ppm_features,
        train_ppm_y,
        params_grid,
        seed_
    )
    
    max_tau_idx = cv_result.val_tau.argmax(axis = 0)
    best_params = cv_result.iloc[max_tau_idx][:3].to_dict()
    
    
    model = train(
        train_x, 
        train_y, 
        int(best_params['seed']),
        best_params['learning_rate'], 
        int(best_params['epochs'])
    )
    
    pred_prob = model.predict(test_x)
    pred = np.argmax(pred_prob, axis = 1)
    
    scc = K.losses.SparseCategoricalCrossentropy()
    train_loss = scc(train_y, model.predict(train_x)).numpy()
    test_loss = scc(test_y, pred_prob).numpy()
      
    result_ = {
        'seed': seed_,
        'parameters': best_params,
        'train_loss': train_loss, 
        'test_loss': test_loss, 
        'precision': precision_score(test_ppm_y, pred, average = 'macro'), 
        'recall': recall_score(test_ppm_y, pred, average = 'macro'), 
        'f1': f1_score(test_ppm_y, pred, average = 'macro'), 
        'accuracy': accuracy_score(test_ppm_y, pred),
        'tau': stats.kendalltau(test_ppm_y, pred).correlation
    }
            
            
    wandb.log({
        'seed': seed_,
        'parameters': best_params,
        'train_loss': train_loss, 
        'test_loss': test_loss,
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
      
pd.DataFrame(result).to_csv('../../test_results/features/ppm_mlp.csv', header = True, index = False)
wandb.finish()


