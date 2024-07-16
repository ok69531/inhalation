import sys
sys.path.append('../')

import os
import logging
import warnings
from tqdm import tqdm
from rdkit import RDLogger 

import numpy as np
import pandas as pd
from scipy.spatial import distance
from imblearn.over_sampling import SMOTE

from module.argument import get_parser
from module.read_data import (
    load_pred_data,
    multiclass2binary,
    new_multiclass2binary
)
from module.smiles2fing import smiles2fing
from module.get_model import load_model
from module.common import (
    load_val_result,
    print_best_param
)

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(format='', level=logging.INFO)


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


def mahalanobis_distance(fingerprint, train_fingerprints, epsilon=1e-10):
    ''' 마할라노비스 거리 계산 함수 '''
    centroid = np.mean(train_fingerprints, axis=0)
    cov_matrix = np.cov(train_fingerprints.T) + np.eye(train_fingerprints.shape[1]) * epsilon
    return distance.mahalanobis(fingerprint, centroid, np.linalg.inv(cov_matrix))


def euclidean_distance(fingerprint, train_fingerprints):
    ''' 유클리드 거리, threshold 계산 함수 '''
    centroid = np.mean(train_fingerprints, axis=0)
    return distance.euclidean(fingerprint, centroid)


def calculate_ad_threshold(train_fingerprints, k=3, Z=0.5):
    """Calculate the Applicability Domain threshold based on Euclidean distances."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train_fingerprints)
    distances, _ = nbrs.kneighbors(train_fingerprints)
    mean_distances = distances[:, 1:].mean(axis=1)  # Exclude the first neighbor (itself)
    y_bar = mean_distances.mean()
    sigma = mean_distances.std()
    D_T = y_bar + Z * sigma
    return D_T


def eval_model_with_ad(model, x_train, x_test, k=3, Z=0.5):
    ''' 최적 모델을 활용하여 화학물질 독성 예측하는 함수 생성 '''
    # Calculate Applicability Domain threshold
    print("Calculating Applicability Domain threshold...")
    D_T = calculate_ad_threshold(x_train, k, Z)
    print(f"Applicability Domain threshold (D_T): {D_T}")

    # Predict target data
    print("Predicting target data...")
    
    euclidean_distances = []
    mahalanobis_distances = []
    euclidean_reliability = []
    mahalanobis_reliability = []
    
    for fp in tqdm(np.array(x_test)):
        md = mahalanobis_distance(fp, x_train)
        ed = euclidean_distance(fp, x_train)
        mahalanobis_distances.append(md)
        euclidean_distances.append(ed)
        
        if ed < D_T:
            euclidean_reliability.append("reliable")
        else:
            euclidean_reliability.append("unreliable")
        
        if md < D_T:
            mahalanobis_reliability.append("reliable")
        else:
            mahalanobis_reliability.append("unreliable")
        
    pred = model.predict(x_test)
    pred_prob = model.predict_proba(x_test)[:, 1]
    
    results = pd.DataFrame({
        'Prediction': pred,
        'Predicted Probability': pred_prob,
        'Euclidean Reliablity': euclidean_reliability, 
        'Euclidean Distance': euclidean_distances,
        'Mahalanobis Reliablity': mahalanobis_reliability, 
        'Mahalanobis Distance': mahalanobis_distances
    })
        
    return results


def main():
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    if (args.tg_num == 403) & (args.inhale_type == 'aerosol'):
        args.model = 'qda'
        args.fp_type = 'maccs'
        args.add_md = True
        args.cat3tohigh = True
        is_smote = True
    elif (args.tg_num == 403) & (args.inhale_type == 'vapour'):
        args.model = 'xgb'
        args.fp_type = 'morgan'
        args.add_md = True
        args.cat3tohigh = True
        is_smote = False
    elif (args.tg_num == 412) & (args.inhale_type == 'aerosol'):
        # rdkit과 rdkit-md의 f1-score가 performance가 동일. auc가 더 높은 rdkit-md 사용
        args.model = 'logistic'
        args.fp_type = 'rdkit'
        args.add_md = True
        args.cat3tohigh = False
        is_smote = True
    elif (args.tg_num == 412) & (args.inhale_type == 'vapour'):
        args.model = 'mlp'
        args.fp_type = 'maccs'
        args.add_md = False
        args.cat3tohigh = False
        is_smote = True
    elif (args.tg_num == 413) & (args.inhale_type == 'aerosol'):
        # lda, dt, gbt 세 모델의 performance가 동일. auc가 제일 높은 dt 사용
        args.model = 'dt'
        args.fp_type = 'maccs'
        args.add_md = True
        args.cat3tohigh = False
        is_smote = True
    elif (args.tg_num == 413) & (args.inhale_type == 'vapour'):
        args.model = 'rf'
        args.fp_type = 'morgan'
        args.add_md = True
        args.cat3tohigh = False
        is_smote = True

    x, y = load_data(path = '../data', args = args)
    if args.cat3tohigh:
        y = new_multiclass2binary(y, args.tg_num)
    else:
        y = multiclass2binary(y, args.tg_num)
    
    fingerprints, pred_df, pred_df_origin = load_pred_data(args)
    
    
    if args.add_md:
        if args.fp_type == 'maccs':
            fp_length = 167
        elif args.fp_type == 'toxprint':
            fp_length = 729
        elif args.fp_type == 'morgan':
            fp_length = 1024
        else:
            fp_length = 2048

        train_descriptors = x.iloc[:, fp_length:]
        descriptors_colnames = train_descriptors.columns
        test_descriptors = fingerprints.iloc[:, fp_length:]
        
        if train_descriptors.shape[1] != test_descriptors.shape[1]:
            drop_col_name = [x for x in descriptors_colnames if x not in test_descriptors.columns]
            train_descriptors = train_descriptors.drop(columns = drop_col_name)
            descriptors_colnames = train_descriptors.columns
        
        logging.info('Number of Descriptors: {}'.format(len(descriptors_colnames)))
        
        scaler = MinMaxScaler()
        scaled_train_descriptors = pd.DataFrame(scaler.fit_transform(train_descriptors, y))
        scaled_train_descriptors.columns = descriptors_colnames
        x = pd.concat([x.iloc[:, :fp_length], scaled_train_descriptors], axis = 1)
        
        scaled_test_descriptors = pd.DataFrame(scaler.transform(fingerprints.iloc[:, fp_length:]))
        scaled_test_descriptors.columns = descriptors_colnames
        fingerprints.iloc[:, fp_length:] = scaled_test_descriptors
    
    
    val_result = load_val_result(path = '..', args = args, is_smote = is_smote)
    best_param = print_best_param(val_result, args.metric)
    
    model = load_model(model = args.model, seed = 0, param = best_param)
    
    if is_smote:
        smote = SMOTE(random_state = args.smoteseed, k_neighbors = args.neighbor)
        x, y = smote.fit_resample(x, y)
    else: 
        pass
    
    model.fit(x, y)
    
    results = eval_model_with_ad(model, x, fingerprints, k = 3, Z = 0.5)
    print(f'number of Euclidean Reliablity: {(results["Euclidean Reliablity"] == "reliable").sum()}')
    print(f'number of Mahalanobis Reliablity: {(results["Mahalanobis Reliablity"] == "reliable").sum()}')
    
    pred_df = pd.concat([pred_df, results], axis = 1)
    result = pd.merge(pred_df_origin, pred_df, how = 'left', on = ('PREFERRED_NAME', 'SMILES'))
    
    save_path = 'pred_result_with_ad/external'
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    result.to_excel(os.path.join(save_path, f'tg{args.tg_num}_{args.inhale_type}_{args.fp_type}_md{args.add_md}_{args.model}.xlsx'), header = True, index = False)


if __name__ == '__main__':
    main()
