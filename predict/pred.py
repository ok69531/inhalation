import sys
sys.path.append('../')

import warnings

import numpy as np
import pandas as pd

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
        args.model = 'dt'
    elif (args.tg_num == 412) & (args.inhale_type == 'vapour'):
        args.model = 'mlp'
    elif (args.tg_num == 412) & (args.inhale_type == 'aerosol'):
        args.model = 'lgb'
    elif (args.tg_num == 413) & (args.inhale_type == 'vapour'):
        args.model = 'plsda'
    elif (args.tg_num == 413) & (args.inhale_type == 'aerosol'):
        args.model = 'dt'
    
    val_result = load_val_result(path = '../', tg_num = args.tg_num, inhale_type = args.inhale_type, model = args.model)
    best_param = print_best_param(val_result, args.metric)
    
    model = load_model(model = args.model, seed = 0, param = best_param)
    
    model.fit(x, y)
    
    if args.model == 'plsda':
        pred_score = model.predict(fingerprints)
        pred = np.where(pred_score < 0.5, 0, 1).reshape(-1)
    else:
        pred = model.predict(fingerprints)
    
    pred_df['pred'] = pred
    result = pd.merge(pred_df_origin, pred_df[['No', 'CasRN', 'Chemical name', 'pred']], how = 'left', on = ('No', 'CasRN', 'Chemical name'))
    result.to_excel(f'pred_result/tg{args.tg_num}_{args.inhale_type}_{args.model}.xlsx', header = True, index = False)


if __name__ == '__main__':
    main()
