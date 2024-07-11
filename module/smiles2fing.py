import numpy as np
import pandas as pd

try: 
    from rdkit import Chem
    from rdkit.Chem import (
        MACCSkeys, 
        AllChem, 
        Descriptors,
        rdMolDescriptors, 
        rdFingerprintGenerator
    )
    # from rdkit.Chem.AllChem import GetRDKitFPGenerator
    from rdkit.ML.Descriptors import MoleculeDescriptors
    
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit-pypi"])
    # subprocess.check_call([sys.executable, "-m", "conda", "install", "rdkit", "-c conda-forge"])
    
    from rdkit import Chem
    from rdkit.Chem import (
        MACCSkeys, 
        AllChem, 
        Descriptors,
        rdMolDescriptors, 
        rdFingerprintGenerator
    )
    # from rdkit.Chem.AllChem import GetRDKitFPGenerator
    from rdkit.ML.Descriptors import MoleculeDescriptors



def smiles2fing(data, args):
    smiles = data.SMILES
    fing_type = args.fp_type
    
    ms_tmp = [Chem.MolFromSmiles(i) for i in smiles]
    ms_none_idx = [i for i in range(len(ms_tmp)) if ms_tmp[i] == None]
    
    ms = list(filter(None, ms_tmp))
    
    if fing_type == 'toxprint':
        sheet_name = 'TG' + str(args.tg_num) + '_' + args.inhale_type.title()
        toxprints = pd.read_excel('./data/toxprint.xlsx', header = 0, sheet_name = sheet_name)
        fingerprints = pd.merge(data.CasRN.rename('INPUT').to_frame(), toxprints, on = 'INPUT')
        fingerprints = fingerprints.iloc[:, 4:]
        
        tp_ms_none_idx = list(fingerprints.iloc[:, 0].index[fingerprints.iloc[:, 0].apply(np.isnan)])
    
    else:
        if fing_type == 'maccs':
            maccs = [MACCSkeys.GenMACCSKeys(i) for i in ms]
            bits = [i.ToBitString() for i in maccs]
        elif fing_type == 'topo':
            topological = [rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(i) for i in ms]
            bits = [i.ToBitString() for i in topological]
        elif fing_type == 'morgan':
            morgen = rdFingerprintGenerator.GetMorganGenerator(radius = 2, fpSize=1024)
            mor_fp = [morgen.GetFingerprint(i) for i in ms]
            bits = [i.ToBitString() for i in mor_fp]
        elif fing_type == 'rdkit':
            rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
            rdkit_fp = [rdkgen.GetFingerprint(i) for i in ms]
            bits = [i.ToBitString() for i in rdkit_fp]
    
        fingerprints = pd.DataFrame({fing_type: bits})
        fingerprints = fingerprints[fing_type].str.split(pat = '', n = len(bits[0]), expand = True)
        fingerprints.drop(fingerprints.columns[0], axis = 1, inplace = True)
    
    colname = [fing_type + '_' + str(i) for i in range(1, fingerprints.shape[1] + 1)]
    fingerprints.columns = colname
    
    if args.add_md:
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        
        if fing_type == 'toxprint':
            fingerprints = fingerprints.drop(ms_none_idx).dropna()
            ms_none_idx.extend(tp_ms_none_idx)
            ms = [ms_tmp[i] for i in range(len(ms_tmp)) if i not in set(ms_none_idx)]
        
        descriptors = pd.DataFrame([dict(zip(descriptor_names, calc.CalcDescriptors(m))) for m in ms]).dropna(axis = 1)
        
        fingerprints = fingerprints.astype(int).reset_index(drop = True)
        fingerprints = pd.concat((fingerprints, descriptors), axis = 1)
        
    else:
        fingerprints = fingerprints.dropna().astype(int).reset_index(drop = True)
        if fing_type == 'toxprint': 
            ms_none_idx = tp_ms_none_idx
    
    return ms_none_idx, fingerprints
