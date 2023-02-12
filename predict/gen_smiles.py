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
