#%%
''' 
    1. LC50 데이터 전처리
        > 부등호, 값, 단위, air, nominal/analytical 구분
        > other이 포함된 행 제거
        > 단위 통일
        > 여러개 있는 값은 어떤 것을 쓸 건지 결정 ,, 보수적으로? 
'''

#%%
import os
import re
import openpyxl

import pandas as pd
import numpy as np 

pd.set_option('mode.chained_assignment', None)


#%%
lc50_tmp = pd.read_excel('tg403_raw_BSY.xlsx', header = 0)

lc50_tmp['Dose descriptor'].unique()
lc50_idx = [i for i in range(len(lc50_tmp)) if len(re.findall('LC50', str(lc50_tmp['Dose descriptor'][i]))) != 0]
lc50_tmp['Dose descriptor'][lc50_idx].unique()


lc50 = lc50_tmp[(lc50_tmp['Dose descriptor'] == 'LC50') | 
                (lc50_tmp['Dose descriptor'] == 'other: approximate LC50') | 
                (lc50_tmp['Dose descriptor'] == 'other: LC50')]
lc50.reset_index(drop = True, inplace = True)
lc50['Dose descriptor'].unique()


# Value에 other: ~~ 이렇게 돼있는거 제거
other_idx = [i for i in range(len(lc50)) if len(re.findall('other', str(lc50['Effect level'][i]))) != 0]
lc50['Effect level'][other_idx] = [re.sub(' other: ', '', lc50['Effect level'][i]) for i in other_idx]


# Value에 ca. ~ 이렇게 돼있는거에서 ca. 제거
ca_idx = [i for i in range(len(lc50)) if len(re.findall('ca\.', str(lc50['Effect level'][i]))) != 0]
lc50['Effect level'][ca_idx] = [re.sub('ca\. ', '', lc50['Effect level'][i]) for i in ca_idx]


# 단위 통일
for i in range(lc50.shape[0]):
    lc50['Effect level'][i] = re.sub('mg/m³|mg/m3', 'mg/m^3', str(lc50['Effect level'][i]))
    lc50['Effect level'][i] = re.sub('g/m3', 'g/m\^3', str(lc50['Effect level'][i]))
    lc50['Effect level'][i] = re.sub('mg/l', 'mg/L', str(lc50['Effect level'][i]))


# Value (value) 에서 괄호 안에 값 제거
np.unique([re.findall('\(.*?\)', str(i)) for i in lc50['Effect level']])
lc50['Effect level'] = [re.sub('\(.*?\)', '', str(i)) for i in lc50['Effect level']]

# Value datframe
val_df = lc50.copy()
val_df['Value_split'] = [val_df['Effect level'][i].split(' ') for i in range(len(val_df))]


# range Value
val_df['range_cat'] = [len(re.findall('-', val_df['Effect level'][i])) for i in range(val_df.shape[0])]

range_idx = val_df['range_cat'][val_df['range_cat'] != 0].index
idx = set(list(range(val_df.shape[0]))) - set(range_idx)
val_df.iloc[range_idx]


#%%
# range가 아닌 value들부터

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


# 부등호
val_df['lower_ineq'] = ''

for i in idx:
    if isFloat(val_df['Value_split'][i][0]):
        val_df['lower_ineq'][i] = np.nan
        
    else:
        val_df['lower_ineq'][i] = val_df['Value_split'][i][0]
        val_df['Value_split'][i].remove(val_df['Value_split'][i][0])


# 값
val_df['lower_value'] = ''

for i in idx:
    if isFloat(''.join(val_df['Value_split'][i][:4])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:4]))
        del val_df['Value_split'][i][:4]
        
    elif isFloat(''.join(val_df['Value_split'][i][:3])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:3]))
        del val_df['Value_split'][i][:3]
    
    elif isFloat(''.join(val_df['Value_split'][i][:2])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:2]))
        del val_df['Value_split'][i][:2]
    
    elif isFloat(''.join(val_df['Value_split'][i][:1])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:1]))
        del val_df['Value_split'][i][:1]


# 단위
# val_df['unit'] = ''
u_ = ['µg/m^3', 'mg/m^3', 'g/m^3', 'mg/L', 'ppm']
val_df['unit'] = val_df['Effect level'].apply(lambda x: ''.join(y for y in x.split() if y in u_))

for i in idx:
    try:
        val_df['Value_split'][i].remove(val_df['unit'][i])
    except ValueError:
        pass
    # try:
        # u_ = re.compile('\u03BCg/m^3|mg/m\^3|g/m\^3|mg/L|ppm')
        # u_ = re.compile('µg/m^3|mg/m\^3|g/m\^3|mg/L|ppm')
        # val_df['unit'][i] = re.findall(u_, val_df['Effect level'][i])[0]
        # val_df['Value_split'][i].remove(re.findall('µg/m^3|g/m\^3|mg/m\^3|mg/L|ppm', val_df['Effect level'][i])[0])

    # except IndexError:
    #     val_df['unit'][i] = np.nan

# air 
val_df['air'] = val_df['Effect level'].apply(lambda x: ''.join(y for y in x.split() if y in ['air']))


# time
val_df['Exp. duration'].unique()

val_df['time'] = ''
val_df['time'][val_df['Exp. duration'].isna()] = np.nan

time_idx = val_df[val_df['Exp. duration'].isna() == False].index

for i in time_idx:
    if isFloat(val_df['Exp. duration'][i]):
        val_df['time'][i] = np.nan
    
    elif val_df['Exp. duration'][i].split(' ')[-1] == 'h':
        val_df['time'][i] = float(val_df['Exp. duration'][i].split(' ')[0])
    
    elif val_df['Exp. duration'][i].split(' ')[-1] == 'min':
        val_df['time'][i] = float(val_df['Exp. duration'][i].split(' ')[0])/60

# for i in idx:
#     try:
#         if val_df['Value_split'][i][0] == 'air':
#             val_df['air'][i] = val_df['Value_split'][i][0]
#             del val_df['Value_split'][i][0]
#         else:
#             val_df['air'][i] = np.nan
#     except IndexError:
#         val_df['air'][i] = np.nan


# nominal / analytical
# for i in idx:
#     try:
#         val_df['nominal/analytical'][i] = re.sub('\(|\)', '', val_df['Value_split'][i][-1])
#         del val_df['Value_split'][i][-1]
#     except IndexError:
#         val_df['nominal/analytical'][i] = np.nan


#%%
# range_idx

# 부등호
val_df['upper_ineq'] = ''
       
for i in range_idx:
    try:
        range_ineq = re.findall('>\=|<\=|>|<', val_df['Effect level'][i])
        val_df['lower_ineq'][i] = range_ineq[0]
        val_df['upper_ineq'][i] = range_ineq[1]
        
        val_df['Value_split'][i] = [j for j in val_df['Value_split'][i] if j not in range_ineq]
        
    except IndexError:
        val_df['lower_ineq'][i] = np.nan
        val_df['upper_ineq'][i] = np.nan


# value
val_df['upper_value'] = ''

for i in range_idx:
    try:
        if val_df['Value_split'][i].index('-') == 1:
            val_df['lower_value'][i] = float(val_df['Value_split'][i][0])
        # val_df['Value_split'][i].remove(val_df['Value_split'][i][0])
        
            if isFloat(''.join(val_df['Value_split'][i][2:4])):
                val_df['upper_value'][i] = float(''.join(val_df['Value_split'][i][2:4]))
                # val_df['Value_split'][i].remove(''.join(val_df['Value_split'][i][2:4]))
            else:
                val_df['upper_value'][i] = float(val_df['Value_split'][i][2])
                # val_df['Value_split'][i].remove(val_df['Value_split'][i][2])
            
        elif val_df['Value_split'][i].index('-') == 2:
            val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:2]))
            val_df['upper_value'][i] = float(''.join(val_df['Value_split'][i][3:5]))
    
    except ValueError:
        val_df['lower_value'][i] = np.nan
        val_df['upper_value'][i] = np.nan
        


# # unit
# for i in range_idx:
#     try:
#         val_df['unit'][i] = re.findall('g/m\^3|mg/m\^3|mg/L|ppm', val_df['Value_tmp'][i])[0]
#         val_df['Value_split'][i] = [j for j in val_df['Value_split'][i] if j not in val_df['unit'][i]]
#     except IndexError:
#         val_df['unit'][i] = np.nan


# # air
# for i in range_idx:
#     try:
#         val_df['air'][i] = re.findall('air', val_df['Value_tmp'][i])[0]
#         val_df['Value_split'][i] = [j for j in val_df['Value_split'][i] if j not in val_df['air'][i]]
#     except IndexError:
#         val_df['air'][i] = np.nan


# # nominal/analytical
# for i in range_idx:
#     tmp = re.findall('\([a-z]*\)', val_df['Value_tmp'][i])
#     try:
#         val_df['nominal/analytical'][i] = re.sub('\(|\)', '', tmp[0])
#     except IndexError:
#         val_df['nominal/analytical'][i] = np.nan


#%%
# SMILES = 0 인 value 변환
from urllib.request import urlopen
from urllib.parse import quote 

def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return '-'

smi_idx = val_df['Final_SMILES'][val_df['Final_SMILES'] == 0].index

smi_ =  [CIRconvert(i) for i in val_df['CasRN'][smi_idx]]
val_df['Final_SMILES'][smi_idx] = smi_


#%%
val_df.iloc[range_idx]
val_df.iloc[4871]


val_df.drop(['Value_split', 'range_cat'], axis = 1, inplace = True)
val_df.to_excel('tg403_lc50.xlsx', header = True, index = False)
