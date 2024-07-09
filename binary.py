import os
import json
import warnings
import logging

import numpy as np
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
    multiclass2binary,
    new_multiclass2binary
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
logging.basicConfig(format='', level=logging.INFO)

def main():
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

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
            model = load_model(model = args.model, seed = seed, param = params[p])
            
            cv_result = binary_cross_validation(model, x_train, y_train, seed)
            
            result['precision']['model'+str(p)].append(cv_result['val_precision'])
            result['recall']['model'+str(p)].append(cv_result['val_recall'])
            result['f1']['model'+str(p)].append(cv_result['val_f1'])
            result['accuracy']['model'+str(p)].append(cv_result['val_accuracy'])
            result['auc']['model'+str(p)].append(cv_result['val_auc'])

    if args.cat3tohigh:
        save_path = f'tg{args.tg_num}_cat3high_val_results/binary/{args.fp_type}_md{args.add_md}'
        if os.path.isdir(save_path):
            pass
        else:
            os.makedirs(save_path)
        json.dump(result, open(f'{save_path}/{args.inhale_type}_{args.model}.json', 'w'))
    else:
        save_path = f'tg{args.tg_num}_val_results/binary/{args.fp_type}_md{args.add_md}'
        if os.path.isdir(save_path):
            pass
        else:
            os.makedirs(save_path)
        json.dump(result, open(f'{save_path}/{args.inhale_type}_{args.model}.json', 'w'))
    
    best_param = print_best_param(val_result = result, metric = args.metric)
    
    m = list(result['model'].keys())[list(result['model'].values()).index(best_param)]
    
    # val result
    precision = result['precision'][m]
    recall = result['recall'][m]
    acc = result['accuracy'][m]
    auc = result['auc'][m]
    f1 = result['f1'][m]
    
    logging.info("best param: {}".format(best_param))
    logging.info("validation result")
    logging.info("precision: {:.3f}({:.3f})".format(np.mean(precision), np.std(precision)))
    logging.info("recall: {:.3f}({:.3f})".format(np.mean(recall), np.std(recall)))
    logging.info("accuracy: {:.3f}({:.3f})".format(np.mean(acc), np.std(acc)))
    logging.info("auc: {:.3f}({:.3f})".format(np.mean(auc), np.std(auc)))
    logging.info("f1: {:.3f}({:.3f})".format(np.mean(f1), np.std(f1)))
    
    # test reulst
    model = load_model(model = args.model, seed = seed, param = best_param)
    
    model.fit(x_train, y_train)
    
    if args.model == 'plsda':
        pred_score = model.predict(x_test)
        pred = np.where(pred_score < 0.5, 0, 1).reshape(-1)
    else:
        pred = model.predict(x_test)
        pred_score = model.predict_proba(x_test)[:, 1]
    
    logging.info("test result")
    logging.info("best param: {}".format(best_param))
    logging.info("precision: {:.3f}".format(precision_score(y_test, pred)))
    logging.info("recall: {:.3f}".format(recall_score(y_test, pred)))
    logging.info("accuracy: {:.3f}".format(accuracy_score(y_test, pred)))
    logging.info("auc: {:.3f}".format(roc_auc_score(y_test, pred_score)))
    logging.info("f1: {:.3f}".format(f1_score(y_test, pred)))


if __name__ == '__main__':
    main()
