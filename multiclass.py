import json
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score
)

from module.argument import get_parser
from module.read_data import load_data
from module.smiles2fing import smiles2fing
from module.get_model import (
    load_model,
    load_hyperparameter
)
from module.common import (
    data_split, 
    multiclass_cross_validation,
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
    
    x_train, x_test, y_train, y_test = data_split(x, y, args.splitseed)

    # cross validation
    params = load_hyperparameter(args.model)

    result = {}
    result['model'] = {}
    result['precision'] = {}
    result['recall'] = {}
    result['f1'] = {}
    result['accuracy'] = {}

    for p in tqdm(range(len(params))):
        
        result['model']['model'+str(p)] = params[p]
        result['precision']['model'+str(p)] = []
        result['recall']['model'+str(p)] = []
        result['f1']['model'+str(p)] = []
        result['accuracy']['model'+str(p)] = []
        
        for seed in range(args.num_run):
            model = load_model(model = args.model, seed = seed, param = params[p])
            
            cv_result = multiclass_cross_validation(model, x_train, y_train, seed)
            
            result['precision']['model'+str(p)].append(cv_result['val_precision'])
            result['recall']['model'+str(p)].append(cv_result['val_recall'])
            result['f1']['model'+str(p)].append(cv_result['val_f1'])
            result['accuracy']['model'+str(p)].append(cv_result['val_accuracy'])

    json.dump(result, open(f'tg{args.tg_num}_val_results/multiclass/{args.inhale_type}_{args.model}.json', 'w'))
    
    
    best_param = print_best_param(val_result = result, metric = args.metric)
    print(best_param)
    
    precision, recall, accuracy, f1 = [], [], [], []
    
    # test reulst
    for seed in range(args.num_run):
        model = load_model(model = args.model, seed = seed, param = best_param)
        
        if args.model == 'plsda':
            y_train = pd.get_dummies(y_train)
        
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        
        if args.model == 'plsda':
            pred = np.argmax(pred, axis = 1)
        
        precision.append(precision_score(y_test, pred, average = 'macro'))
        recall.append(recall_score(y_test, pred, average = 'macro'))
        accuracy.append(accuracy_score(y_test, pred))
        f1.append(f1_score(y_test, pred, average = 'macro'))
        
    print(f'================================= \
          \ntg{args.tg_num} {args.inhale_type} {args.model} \
          \nprecision: {np.mean(precision):.3f}({np.std(precision):.3f}) \
          \nrecall: {np.mean(recall):.3f}({np.std(recall):.3f}) \
          \naccuracy: {np.mean(accuracy):.3f}({np.std(accuracy):.3f}) \
          \nf1: {np.mean(f1):.3f}({np.std(f1):.3f})')


if __name__ == '__main__':
    main()
