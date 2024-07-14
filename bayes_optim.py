import os
import json
import logging
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    make_scorer
)
from sklearn.preprocessing import MinMaxScaler

from module.argument import get_parser
from module.read_data import (
    load_data,
    multiclass2binary,
    new_multiclass2binary
)
from module.smiles2fing import smiles2fing
from module.common import (
    data_split, 
    print_best_param
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings('ignore')
logging.basicConfig(format='', level=logging.INFO)



def main():
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])


    if (args.tg_num == 403) & (args.inhale_type == 'aerosol'):
        args.model = 'qda'
        args.fp_type = 'maccs'
        args.add_md = True
        args.cat3tohigh = True
    elif (args.tg_num == 403) & (args.inhale_type == 'vapour'):
        args.model = 'xgb'
        args.fp_type = 'morgan'
        args.add_md = True
        args.cat3tohigh = True
    elif (args.tg_num == 412) & (args.inhale_type == 'aerosol'):
        # rdkit과 rdkit-md의 f1-score가 performance가 동일. auc가 더 높은 rdkit-md 사용
        args.model = 'logistic'
        args.fp_type = 'rdkit'
        args.add_md = True
        args.cat3tohigh = False
    elif (args.tg_num == 412) & (args.inhale_type == 'vapour'):
        args.model = 'mlp'
        args.fp_type = 'maccs'
        args.add_md = False
        args.cat3tohigh = False
    elif (args.tg_num == 413) & (args.inhale_type == 'aerosol'):
        # lda, dt, gbt 세 모델의 performance가 동일. auc가 제일 높은 dt 사용
        args.model = 'dt'
        args.fp_type = 'maccs'
        args.add_md = True
        args.cat3tohigh = False
    elif (args.tg_num == 413) & (args.inhale_type == 'vapour'):
        args.model = 'rf'
        args.fp_type = 'morgan'
        args.add_md = True
        args.cat3tohigh = False

    logging.info('=================================')
    logging.info('tg{} {} {}'.format(args.tg_num, args.inhale_type, args.model))
    logging.info('Fingerprints: {}, Use Descriptors: {}'.format(args.fp_type, args.add_md))


    x, y = load_data(path = 'data', args = args)
    if args.cat3tohigh:
        y = new_multiclass2binary(y, args.tg_num)
    else:
        y = multiclass2binary(y, args.tg_num)

    x_train, x_test, y_train, y_test = data_split(x, y, args.splitseed)

    if args.add_md:
        if args.fp_type == 'maccs':
            fp_length = 167
        elif args.fp_type == 'toxprint':
            fp_length = 729
        elif args.fp_type == 'morgan':
            fp_length = 1024
        else:
            fp_length = 2048

        train_descriptors = x_train.iloc[:, fp_length:]
        descriptors_colnames = train_descriptors.columns
        
        logging.info('Number of Descriptors: {}'.format(len(descriptors_colnames)))
        
        scaler = MinMaxScaler()
        scaled_train_descriptors = pd.DataFrame(scaler.fit_transform(train_descriptors, y_train))
        scaled_train_descriptors.columns = descriptors_colnames
        x_train.iloc[:, fp_length:] = scaled_train_descriptors

        scaled_test_descriptors = pd.DataFrame(scaler.transform(x_test.iloc[:, fp_length:]))
        scaled_test_descriptors.columns = descriptors_colnames
        x_test.iloc[:, fp_length:] = scaled_test_descriptors

    if (args.tg_num == 403) & (args.inhale_type == 'vapour'):
        pipeline = Pipeline([
            ('classifier', load_model(args.model))
        ])
    else:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state = args.smoteseed, k_neighbors = args.neighbor)),
            ('classifier', load_model(args.model))
        ])

    best_params_list = []

    for seed in tqdm(range(args.num_run)):
        search_space = get_search_space(args.model, seed)
        
        opt = BayesSearchCV(
            estimator = pipeline,
            search_spaces = search_space,
            cv = 5,
            n_iter = 500,
            random_state = seed,
            scoring=make_scorer(f1_score)
        )
        opt.fit(x_train, y_train)
            
        best_params_list.append(dict(opt.best_params_.items()))

    test_prec_list = []
    test_rec_list = []
    test_f1_list = []
    test_acc_list = []
    test_auc_list = []

    for i in range(len(best_params_list)):
        best_params = dict(best_params_list[i])
        best_params = {k.split('__')[-1]: v for k, v in best_params.items()}
        
        smote = SMOTE(random_state = args.smoteseed, k_neighbors = args.neighbor)
        x_train, y_train = smote.fit_resample(x_train, y_train)
        
        model = load_model(args.model, **best_params)
        model.fit(x_train, y_train)
        
        pred = model.predict(x_test)
        pred_prob = model.predict_proba(x_test)[:, 1]
        
        test_prec_list.append(precision_score(y_test, pred))
        test_rec_list.append(recall_score(y_test, pred))
        test_f1_list.append(f1_score(y_test, pred))
        test_acc_list.append(accuracy_score(y_test, pred))
        test_auc_list.append(roc_auc_score(y_test, pred_prob))

    checkpoints = {
        'params': best_params_list,
        'precision': test_prec_list,
        'recall': test_rec_list,
        'accuracy': test_acc_list,
        'auc': test_acc_list
    }
    
    logging.info("best param: {}".format(best_params_list))
    logging.info("precisions: {}".format(test_prec_list))
    logging.info("recalls: {}".format(test_rec_list))
    logging.info("accuracies: {}".format(test_acc_list))
    logging.info("aucs: {}".format(test_auc_list))
    logging.info("f1s: {}".format(test_f1_list))
    
    logging.info("test result")
    logging.info("precision: {:.3f}({:.3f})".format(np.mean(test_prec_list), np.std(test_prec_list)))
    logging.info("recall: {:.3f}({:.3f})".format(np.mean(test_rec_list), np.std(test_rec_list)))
    logging.info("accuracy: {:.3f}({:.3f})".format(np.mean(test_acc_list), np.std(test_acc_list)))
    logging.info("auc: {:.3f}({:.3f})".format(np.mean(test_auc_list), np.std(test_auc_list)))
    logging.info("f1: {:.3f}({:.3f})".format(np.mean(test_f1_list), np.std(test_f1_list)))
    
    save_path = 'bayes_optim/'
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    json.dump(checkpoints, open(os.path.join(save_path, f'tg{args.tg_num}_{args.inhale_type}_{args.fp_type}_md{args.add_md}.json'), 'w'))


def load_model(model: str, **kwargs):
    if model == 'logistic':
        return LogisticRegression(**kwargs)
        
    elif model == 'dt':
        return DecisionTreeClassifier(**kwargs)
    
    elif model == 'rf':
        return RandomForestClassifier(**kwargs)
    
    elif model == 'xgb':
        return XGBClassifier(**kwargs)
    
    elif model == 'qda':
        return QuadraticDiscriminantAnalysis(**kwargs)
    
    elif model == 'mlp':
        return MLPClassifier(**kwargs)


def get_search_space(model_name, seed):
    if model_name == 'logistic':
        params = {
            'classifier__random_state': [seed],
            'classifier__C': (0.1, 100),
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
    elif model_name == 'dt':
        params = {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': Integer(1, 5),
            'classifier__min_samples_split': Integer(2, 5),
            'classifier__min_samples_leaf': Integer(1, 5)
        }
    elif model_name == 'rf':
        params = {
            'classifier__n_estimators': Integer(5, 150),
            'classifier__criterion': ['gini'],
            'classifier__min_samples_split': Integer(2, 5),
            'classifier__min_samples_leaf': Integer(1, 3),
            'classifier__max_features': ['sqrt', 'log2']
        }
    elif model_name == 'xgb':
        params = {
            'classifier__min_child_weight': Integer(1, 5),
            'classifier__max_depth': Integer(3, 9), 
            'classifier__gamma': (0, 3),
            'classifier__booster': ['gbtree']
        }
    elif model_name == 'qda':
        params = {
            'classifier__reg_param': (0, 1),
            'classifier__tol': (1e-5, 1e-3)
        }
    elif model_name == 'mlp':
        params = {
            'classifier__hidden_layer_sizes': Integer(30, 200),
            'classifier__activation': ['relu', 'tanh'],
            'classifier__solver': ['adam', 'sgd'],
            'classifier__alpha': (1e-4, 1e-2),
            'classifier__learning_rate_init': (1e-3, 1e-1),
            'classifier__max_iter': Integer(50, 200)
        }
    
    return params


if __name__ == '__main__':
    main()
