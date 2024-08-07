import pandas as pd
from .smiles2fing import smiles2fing


def load_data(path: str, args):
    tg_num = args.tg_num
    inhale_type = args.inhale_type
    
    try:
        df = pd.read_excel(f'{path}tg{tg_num}_{inhale_type}.xlsx')
    except:
        df = pd.read_excel(f'{path}/tg{tg_num}_{inhale_type}.xlsx')
    
    drop_idx, fingerprints = smiles2fing(df[['CasRN', 'SMILES']], args)
    
    y = df.category.drop(drop_idx).reset_index(drop = True)
    
    return fingerprints, y


def multiclass2binary(y, tg_num: int):
    '''
        category 0, 1 -> 1
        category 2, 3, 4 -> 0
    '''
    bin_y = y.copy()
    
    if tg_num == 403:
        bin_y[y<2] = 1
        bin_y[y>=2] = 0
    
    else:
        bin_y[y<1] = 1
        bin_y[y>=1] = 0
    
    return bin_y


def new_multiclass2binary(y, tg_num: int):
    '''
        category 0, 1, 2 -> 1
        category 3, 4 -> 0
    '''
    bin_y = y.copy()
    
    if tg_num == 403:
        bin_y[y<3] = 1
        bin_y[y>=3] = 0
    
    else:
        bin_y[y<1] = 1
        bin_y[y>=1] = 0
    
    return bin_y


def load_pred_data(args):
    df_tmp = pd.read_excel('pred_data.xlsx').drop_duplicates(subset = ('PREFERRED_NAME', 'SMILES'))
    
    # try:
    #     df = pd.read_excel(f'{path}/pred_data.xlsx')
    # except: 
    #     df = pd.read_excel(f'{path}pred_data.xlsx')
    
    df = df_tmp[df_tmp['SMILES'].notna()].reset_index(drop = True)
    
    drop_idx, fingerprints = smiles2fing(df, args)
    df = df.drop(drop_idx).reset_index(drop = True)
    
    return fingerprints, df, df_tmp
