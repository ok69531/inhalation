import cirpy
import pandas as pd
from tqdm import tqdm


df = pd.read_excel('pred_data.xlsx', header = 0)
df['SMILES'] = ''

for i in tqdm(range(df.shape[0])):
    try:
        df['SMILES'][i] = cirpy.resolve(df.CasRN[i], 'smiles')
    except:
        pass

# tqdm.pandas()
# df['SMILES'] = df.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))

# df.SMILES.isna().sum()
# df = df[df['SMILES'].notna()].reset_index(drop = True)

df.to_excel('pred_data.xlsx', header = True, index = False)


#
# vapour403 = pd.read_excel('../data/tg403_vapour.xlsx')
# aerosol403 = pd.read_excel('../data/tg403_aerosol.xlsx')
# vapour412 = pd.read_excel('../data/tg412_vapour.xlsx')
# aerosol412 = pd.read_excel('../data/tg412_aerosol.xlsx')
# vapour413 = pd.read_excel('../data/tg413_vapour.xlsx')
# aerosol413 = pd.read_excel('../data/tg413_aerosol.xlsx')

# sum(df.CasRN.isin(vapour403.CasRN))
# sum(df.CasRN.isin(aerosol403.CasRN))
# sum(df.CasRN.isin(vapour412.CasRN))
# sum(df.CasRN.isin(aerosol412.CasRN))
# sum(df.CasRN.isin(vapour413.CasRN))
# sum(df.CasRN.isin(aerosol413.CasRN))

# df['412vapour_train'] = ''
# df['412vapour_train'][df.CasRN.isin(vapour412.CasRN)] = 'o'

# df.to_excel('is_train.xlsx', header = True, index = False)