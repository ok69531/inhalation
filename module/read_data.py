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
    if tg_num == 403:
        y = y.where(y>=2, 0).where(y<2, 1)
    
    else:
        y = y.where(y>=1, 0).where(y<1, 1)
    
    return y
