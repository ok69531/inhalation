import sys
sys.path.append('../')

import warnings
from rdkit import RDLogger 

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from module.argument import get_parser
from module.read_data import (
    load_data,
    load_pred_data,
    multiclass2binary
)
from module.smiles2fing import smiles2fing
from module.get_model import load_model
from module.common import (
    load_val_result,
    print_best_param
)


warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')


def main():
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    x, y = load_data(path = '../data', tg_num = args.tg_num, inhale_type = args.inhale_type)
    y = multiclass2binary(y, args.tg_num)
    
    fingerprints, pred_df, pred_df_origin = load_pred_data()
    
    if (args.tg_num == 403) & (args.inhale_type == 'vapour'):
        args.model = 'lgb'
    elif (args.tg_num == 403) & (args.inhale_type == 'aerosol'):
        args.model = 'rf'
    elif (args.tg_num == 412) & (args.inhale_type == 'vapour'):
        args.model = 'mlp'
    elif (args.tg_num == 412) & (args.inhale_type == 'aerosol'):
        args.model = 'qda'
        args.smoteseed = 0
    elif (args.tg_num == 413) & (args.inhale_type == 'vapour'):
        args.model = 'lda'
        args.smoteseed = 119
    elif (args.tg_num == 413) & (args.inhale_type == 'aerosol'):
        args.model = 'mlp'
    
    val_result = load_val_result(path = '..', is_smote = True, args)
    best_param = print_best_param(val_result, args.metric)
    
    model = load_model(model = args.model, seed = 0, param = best_param)
    
    smote = SMOTE(random_state = args.smoteseed, k_neighbors = args.neighbor)
    x, y = smote.fit_resample(x, y)
    
    model.fit(x, y)
    
    if args.model == 'plsda':
        pred_score = model.predict(fingerprints)
        # pred = np.where(pred_score < 0.5, 0, 1).reshape(-1)
    else:
        pred_score = model.predict_proba(fingerprints)[:, 1]
    
    pred_df['pred'] = pred_score
    result = pd.merge(pred_df_origin, pred_df[['PREFERRED_NAME', 'SMILES', 'pred']], how = 'left', on = ('PREFERRED_NAME', 'SMILES'))
    result.to_excel(f'pred_result/tg{args.tg_num}_{args.inhale_type}_{args.model}.xlsx', header = True, index = False)


if __name__ == '__main__':
    main()
