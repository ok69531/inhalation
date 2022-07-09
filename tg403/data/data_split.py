#%%
import random
import openpyxl
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import column 

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore")


#%%
'''
    data split
'''

data = pd.read_excel('tg403_lc50.xlsx')


len(data['CasRN'].unique())

data['unit'].unique()
data['unit'].isna().sum()
data = data[data['unit'].notna()]
data = data[data['lower_value'].notna()]
data = data[data['Final_SMILES'].notna()]

casrn_na_idx = data[data['CasRN'] == '-'].index

data = data.drop(casrn_na_idx).reset_index(drop = True)


#%%
# ppm data
lc50_ppm_tmp = data[data['unit'] == 'ppm']
lc50_ppm = lc50_ppm_tmp.groupby(['CasRN', 'Final_SMILES'])['lower_value', 'time'].mean().reset_index()
lc50_ppm.columns = ['CasRN', 'SMILES', 'value', 'time']
lc50_ppm['value'].describe()

# mg/L data
lc50_mgl_tmp = data[data['unit'] != 'ppm']
lc50_mgl_tmp['value'] = [lc50_mgl_tmp['lower_value'][i] if lc50_mgl_tmp['unit'][i] == 'mg/L' 
                         else lc50_mgl_tmp['lower_value'][i]*0.001 if lc50_mgl_tmp['unit'][i] == 'mg/m^3' 
                         else lc50_mgl_tmp['lower_value'][i]*0.000001 for i in lc50_mgl_tmp.index]
lc50_mgl = lc50_mgl_tmp.groupby(['CasRN', 'Final_SMILES'])['value', 'time'].mean().reset_index()
lc50_mgl['value'].describe()


# lc50_ppm.to_excel('ppm.xlsx', header = True, index = False)
# lc50_mgl.to_excel('mgl.xlsx', header = True, index = False)


#%%
# random.seed(0)
# train_mgl_idx = random.sample(list(lc50_mgl.index), int(len(lc50_mgl) * 0.8))
# test_mgl_idx = list(set(lc50_mgl.index) - set(train_mgl_idx))

# train_mgl = lc50_mgl.iloc[train_mgl_idx].reset_index(drop = True)
# test_mgl = lc50_mgl.iloc[test_mgl_idx].reset_index(drop = True)

# train_mgl.to_excel('train_mgl.xlsx', header = True, index = False)
# test_mgl.to_excel('test_mgl.xlsx', header = True, index = False)



# train_ppm_idx = random.sample(list(lc50_ppm.index), int(len(lc50_ppm) * 0.8))
# test_ppm_idx = list(set(lc50_ppm.index) - set(train_ppm_idx))

# train_ppm = lc50_ppm.iloc[train_ppm_idx].reset_index(drop = True)
# test_ppm = lc50_ppm.iloc[test_ppm_idx].reset_index(drop = True)

# train_ppm.to_excel('train_ppm.xlsx', header = True, index = False)
# test_ppm.to_excel('test_ppm.xlsx', header = True, index = False)
