import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import sem

from tqdm import tqdm

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score
)

from module.argument import get_parser
from module.read_data import (
    load_data,
    multiclass2binary
)
from module.smiles2fing import smiles2fing
from module.get_model import (
    load_model,
    load_hyperparameter
)
from module.common import (
    data_split, 
    binary_cross_validation,
    print_best_param
)


warnings.filterwarnings('ignore')


def main():
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    x, y = load_data(path = 'data', tg_num = args.tg_num, inhale_type = args.inhale_type)
    y = multiclass2binary(y, args.tg_num)

    # cross validation
    params = load_hyperparameter(args.model)

    result = {}
    result['model'] = {}
    result['precision'] = {}
    result['recall'] = {}
    result['f1'] = {}
    result['accuracy'] = {}
    result['auc'] = {}

    for p in tqdm(range(len(params))):
        
        result['model']['model'+str(p)] = params[p]
        result['precision']['model'+str(p)] = []
        result['recall']['model'+str(p)] = []
        result['f1']['model'+str(p)] = []
        result['accuracy']['model'+str(p)] = []
        result['auc']['model'+str(p)] = []
        
        for seed in range(args.num_run):
            x_train, x_test, y_train, y_test = data_split(x, y, seed)
            
            model = load_model(model = args.model, seed = seed, param = params[p])
            
            cv_result = binary_cross_validation(model, x_train, y_train, seed)
            
            result['precision']['model'+str(p)].append(cv_result['val_precision'])
            result['recall']['model'+str(p)].append(cv_result['val_recall'])
            result['f1']['model'+str(p)].append(cv_result['val_f1'])
            result['accuracy']['model'+str(p)].append(cv_result['val_accuracy'])
            result['auc']['model'+str(p)].append(cv_result['val_auc'])

    json.dump(result, open(f'tg{args.tg_num}_val_results/binary/{args.inhale_type}_{args.model}.json', 'w'))
    
    
    best_param = print_best_param(val_result = result, metric = args.metric)
    precision, recall, accuracy, f1, auc = [], [], [], [], []
    
    # test reulst
    for seed in range(args.num_run):
        x_train, x_test, y_train, y_test = data_split(x, y, seed)
        
        model = load_model(model = args.model, seed = seed, param = best_param)
        
        model.fit(x_train, y_train)
        
        if args.model == 'plsda':
            pred_score = model.predict(x_test)
            pred = np.where(pred_score < 0.5, 0, 1).reshape(-1)
        else:
            pred = model.predict(x_test)
            pred_score = model.predict_proba(x_test)
        
        precision.append(precision_score(y_test, pred))
        recall.append(recall_score(y_test, pred))
        accuracy.append(accuracy_score(y_test, pred))
        auc.append(roc_auc_score(y_test, pred_score))
        f1.append(f1_score(y_test, pred))
        
    print(f'================================= \
          \ntg{args.tg_num} {args.inhale_type} {args.model} \
          \nprecision: {np.mean(precision):.3f}({sem(precision):.3f}) \
          \nrecall: {np.mean(recall):.3f}({sem(recall):.3f}) \
          \naccuracy: {np.mean(accuracy):.3f}({sem(accuracy):.3f}) \
          \nauc: {np.mean(auc):.3f}({sem(auc):.3f}) \
          \nf1: {np.mean(f1):.3f}({sem(f1):.3f})')


if __name__ == '__main__':
    main()
