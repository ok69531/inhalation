import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

from sklearn.metrics import f1_score

from module.argument import get_parser
from module.read_data import (
    load_data,
    multiclass2binary
)
from module.get_model import (
    load_model
)
from module.common import (
    data_split, 
    load_val_result,
    print_best_param
)

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


warnings.filterwarnings('ignore')


def main():
    tg_nums = [403, 412, 413]
    inhale_types = ['vapour', 'aerosol']
    model_names = ['logistic', 'lda', 'qda', 'plsda', 'dt', 'rf', 'gbt', 'xgb', 'lgb', 'mlp']
    thresholds = np.round(np.arange(0.1, 1, 0.1), 1).tolist()
    metric = 'f1'
    
    comb = list(product(tg_nums, inhale_types, model_names, thresholds))
    # data_comb = list(product(tg_nums, inhale_types))
    
    f1_dict = {x: {y: {z: {}.fromkeys(thresholds) for z in model_names} for y in inhale_types} for x in tg_nums}
    
    for c in tqdm(comb):
        tg_num, inhale_type, model_name, thres = c
        
        x, y = load_data(path = 'data', tg_num = tg_num, inhale_type = inhale_type)
        y = multiclass2binary(y, tg_num)
    
        result = load_val_result('', tg_num, inhale_type, model_name)
        best_param = print_best_param(val_result = result, metric = metric)
    
        # test reulst
        f1 = []

        for seed in range(10):
            x_train, x_test, y_train, y_test = data_split(x, y, seed)
            
            model = load_model(model = model_name, seed = seed, param = best_param)
            model.fit(x_train, y_train)
            if model_name == 'plsda':
                pred_score = model.predict(x_test)
            else:
                pred_score = model.predict_proba(x_test)[:, 1]
            pred = np.where(pred_score < thres, 0, 1).reshape(-1)
            
            f1.append(f1_score(y_test, pred))
        
            f1_dict[tg_num][inhale_type][model_name][thres] = {'mean': np.mean(f1), 'std': np.std(f1)}
    
    return f1_dict

f1_dict = main()
model_names = ['logistic', 'lda', 'qda', 'plsda', 'dt', 'rf', 'gbt', 'xgb', 'lgb', 'mlp']


#
vapor403 = pd.DataFrame(f1_dict[403]['vapour'])
vapor403 = pd.concat([vapor403[x].apply(pd.Series) for x in list(vapor403)], 1, keys=list(vapor403))

plt.figure(figsize = (10, 5))
for m in model_names:
    plt.plot(vapor403.index, vapor403[m]['mean'], '-o', markersize='3')
    # plt.errorbar(vapor403.index, vapor403[m]['mean'], yerr = vapor403[m]['std'], capsize = 4, fmt = '-o', markersize='3')
plt.ylim((0, 1))
plt.legend(model_names, ncol = 2)
plt.title('Actue Vapor F1 score')
plt.show()
plt.close()

m = 'logistic'
plt.figure(figsize = (10, 5))
plt.errorbar(vapor403.index, vapor403[m]['mean'], yerr = vapor403[m]['std'], capsize = 4, fmt = '-o', markersize='3')
plt.ylim((0, 1))
plt.title('Actue Vapor Logistic Regression F1 score')
plt.show()
plt.close()


#
aerosol403 = pd.DataFrame(f1_dict[403]['aerosol'])
aerosol403 = pd.concat([aerosol403[x].apply(pd.Series) for x in list(aerosol403)], 1, keys = list(aerosol403))

plt.figure(figsize = (10, 5))
for m in model_names:
    plt.plot(aerosol403.index, aerosol403[m]['mean'], '-o', markersize='3')
    # plt.errorbar(vapor403.index, vapor403[m]['mean'], yerr = vapor403[m]['std'], capsize = 4, fmt = '-o', markersize='3')
plt.ylim((0, 1))
plt.legend(model_names, ncol = 2)
plt.title('Actue Aerosol F1 score')
plt.show()
plt.close()


#
vapor412 = pd.DataFrame(f1_dict[412]['vapour'])
vapor412 = pd.concat([vapor412[x].apply(pd.Series) for x in list(vapor412)], 1, keys = list(vapor412))

plt.figure(figsize = (10, 5))
for m in model_names:
    plt.plot(vapor412.index, vapor412[m]['mean'], '-o', markersize='3')
plt.ylim((0, 1))
plt.legend(model_names, ncol = 2)
plt.title('Sub-Actue Vapor F1 score')
plt.show()
plt.close()


#
aerosol412 = pd.DataFrame(f1_dict[412]['aerosol'])
aerosol412 = pd.concat([aerosol412[x].apply(pd.Series) for x in list(aerosol412)], 1, keys = list(aerosol412))

plt.figure(figsize = (10, 5))
for m in model_names:
    plt.plot(aerosol412.index, aerosol412[m]['mean'], '-o', markersize='3')
    # plt.errorbar(aerosol412.index, aerosol412[m]['mean'], yerr = aerosol412[m]['std'], capsize = 4, fmt = '-o', markersize='3')
plt.ylim((0, 1))
plt.legend(model_names, ncol = 2)
plt.title('Sub-Actue Aerosol F1 score')
plt.show()
plt.close()


#
vapor413 = pd.DataFrame(f1_dict[413]['vapour'])
vapor413 = pd.concat([vapor413[x].apply(pd.Series) for x in list(vapor413)], 1, keys = list(vapor413))

plt.figure(figsize = (10, 5))
for m in model_names:
    plt.plot(vapor413.index, vapor413[m]['mean'], '-o', markersize='3')
plt.ylim((0, 1))
plt.legend(model_names, ncol = 2)
plt.title('Sub-Chronic Vapor F1 score')
plt.show()
plt.close()


#
aerosol413 = pd.DataFrame(f1_dict[413]['aerosol'])
aerosol413 = pd.concat([aerosol413[x].apply(pd.Series) for x in list(aerosol413)], 1, keys = list(aerosol413))

plt.figure(figsize = (10, 5))
for m in model_names:
    plt.plot(aerosol413.index, aerosol413[m]['mean'], '-o', markersize='3')
plt.ylim((0, 1))
plt.legend(model_names, ncol = 2)
plt.title('Sub-Chronic Aerosol F1 score')
plt.show()
plt.close()
