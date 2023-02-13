import pandas as pd
from .smiles2fing import smiles2fing


def load_data(path: str, tg_num: int, inhale_type: str):
    try:
        df = pd.read_excel(f'{path}tg{tg_num}_{inhale_type}.xlsx')
    except:
        df = pd.read_excel(f'{path}/tg{tg_num}_{inhale_type}.xlsx')
    
    drop_idx, fingerprints = smiles2fing(df.SMILES)
    
    y = df.category.drop(drop_idx).reset_index(drop = True)
    
    return fingerprints, y


def multiclass2binary(y, tg_num: int):
    bin_y = y.copy()
    
    if tg_num == 403:
        bin_y[y<2] = 1
        bin_y[y>=2] = 0
    
    else:
        bin_y[y<1] = 1
        bin_y[y>=1] = 0
    
    return bin_y


def load_pred_data():
    df_tmp = pd.read_excel('pred_data.xlsx')
    
    # try:
    #     df = pd.read_excel(f'{path}/pred_data.xlsx')
    # except: 
    #     df = pd.read_excel(f'{path}pred_data.xlsx')
    
    df = df_tmp[df_tmp['SMILES'].notna()].reset_index(drop = True)
    
    drop_idx, fingerprints = smiles2fing(df.SMILES)
    df = df.drop(drop_idx).reset_index(drop = True)
    
    return fingerprints, df, df_tmp
