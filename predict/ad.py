#%%
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

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score
)

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
    
    return df, drop_idx, fingerprints, y


def data_split(X, y, seed):
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = seed)
    
    for train_idx, test_idx in sss.split(X, y):
        train_x = X.iloc[train_idx].reset_index(drop = True)
        train_y = y.iloc[train_idx].reset_index(drop = True)
        test_x = X.iloc[test_idx].reset_index(drop = True)
        test_y = y.iloc[test_idx].reset_index(drop = True)
    
    return train_x, test_x, train_y, test_y, train_idx, test_idx


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


def eval_model_with_ad(model, x_train, x_test, y_test, k=3, Z=0.5):
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
    
    eu_rel_idx = np.where(results['Euclidean Reliablity'] == 'reliable')[0]
    ma_rel_idx = np.where(results['Mahalanobis Reliablity'] == 'reliable')[0]
    
    logging.info("")
    logging.info("Euclidean reliable test result")
    if len(eu_rel_idx) != 0:
        logging.info("precision: {:.3f}".format(precision_score(y_test[eu_rel_idx], pred[eu_rel_idx])))
        logging.info("recall: {:.3f}".format(recall_score(y_test[eu_rel_idx], pred[eu_rel_idx])))
        logging.info("accuracy: {:.3f}".format(accuracy_score(y_test[eu_rel_idx], pred[eu_rel_idx])))
        if len(y_test[eu_rel_idx].unique() == 1):
            logging.info(f"Euclidean reliable data has only {y_test[eu_rel_idx].unique()[0]} label.")
        else:
            logging.info("auc: {:.3f}".format(roc_auc_score(y_test[eu_rel_idx], pred_prob[eu_rel_idx])))
        logging.info("f1: {:.3f}".format(f1_score(y_test[eu_rel_idx], pred[eu_rel_idx])))
    else:
        logging.info("There is no Euclidean reliable data!")
    
    logging.info("")
    logging.info("Mahalanobis reliable test result")
    if len(ma_rel_idx) != 0:
        logging.info("precision: {:.3f}".format(precision_score(y_test[ma_rel_idx], pred[ma_rel_idx])))
        logging.info("recall: {:.3f}".format(recall_score(y_test[ma_rel_idx], pred[ma_rel_idx])))
        logging.info("accuracy: {:.3f}".format(accuracy_score(y_test[ma_rel_idx], pred[ma_rel_idx])))
        if len(y_test[ma_rel_idx].unique() == 1):
            logging.info(f"Euclidean reliable data has only {y_test[ma_rel_idx].unique()[0]} label.")
        else:
            logging.info("auc: {:.3f}".format(roc_auc_score(y_test[ma_rel_idx], pred_prob[ma_rel_idx])))
        logging.info("f1: {:.3f}".format(f1_score(y_test[ma_rel_idx], pred[ma_rel_idx])))
    else:
        logging.info("There is no Mahalanobis reliable data!")
        
    return results


#%%
parser = get_parser()
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

# args.tg_num = 403
# args.inhale_type = 'aerosol'

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

logging.info('=================================')
logging.info('tg{} {} {}'.format(args.tg_num, args.inhale_type, args.model))
logging.info('Fingerprints: {}, Use Descriptors: {}'.format(args.fp_type, args.add_md))

df, drop_idx, x, y = load_data(path = '../data', args = args)
if args.cat3tohigh:
    y = new_multiclass2binary(y, args.tg_num)
else:
    y = multiclass2binary(y, args.tg_num)

x_train, x_test, y_train, y_test, train_idx, test_idx = data_split(x, y, args.splitseed)

test_df = df.drop(drop_idx).reset_index(drop = True).iloc[test_idx].reset_index(drop = True)


if args.add_md:
    if args.fp_type == 'maccs':
        fp_length = 167
    elif args.fp_type == 'toxprint':
        fp_length = 729
    elif args.fp_type == 'morgan':
        fp_length = 1024
    else:
        fp_length = 2048

    train_descriptors = x_train.iloc[:, fp_length:]
    descriptors_colnames = train_descriptors.columns
    
    logging.info('Number of Descriptors: {}'.format(len(descriptors_colnames)))
    
    scaler = MinMaxScaler()
    scaled_train_descriptors = pd.DataFrame(scaler.fit_transform(train_descriptors, y_train))
    scaled_train_descriptors.columns = descriptors_colnames
    x_train.iloc[:, fp_length:] = scaled_train_descriptors

    scaled_test_descriptors = pd.DataFrame(scaler.transform(x_test.iloc[:, fp_length:]))
    scaled_test_descriptors.columns = descriptors_colnames
    x_test.iloc[:, fp_length:] = scaled_test_descriptors

val_result = load_val_result(path = '..', args = args, is_smote = is_smote)
best_param = print_best_param(val_result, args.metric)

model = load_model(model = args.model, seed = 0, param = best_param)

if is_smote:
    smote = SMOTE(random_state = args.smoteseed, k_neighbors = args.neighbor)
    x_train, y_train = smote.fit_resample(x_train, y_train)
else:
    pass

model.fit(x_train, y_train)

result_with_ad = eval_model_with_ad(model, x_train, x_test, y_test, k = 3, Z = 0.5)
result_with_ad = pd.concat([test_df['SMILES'], result_with_ad], axis = 1)

save_path = 'pred_result_with_ad/test'
if os.path.exists(save_path):
    pass
else:
    os.makedirs(save_path)
    
save_path = os.path.join(save_path, f'tg{args.tg_num}_{args.inhale_type}_{args.fp_type}_md{args.add_md}_{args.model}.xlsx')
result_with_ad.to_excel(save_path, header = True, index = False)


# #%%
# from rdkit import Chem
# from rdkit.Chem import MACCSkeys, DataStructs


# args.tg_num = 403
# args.inhale_type = 'aerosol'

# if (args.tg_num == 403) & (args.inhale_type == 'aerosol'):
#     args.model = 'qda'
#     args.fp_type = 'maccs'
#     args.add_md = True
#     args.cat3tohigh = True
#     is_smote = True
# elif (args.tg_num == 403) & (args.inhale_type == 'vapour'):
#     args.model = 'xgb'
#     args.fp_type = 'morgan'
#     args.add_md = True
#     args.cat3tohigh = True
#     is_smote = False
# elif (args.tg_num == 412) & (args.inhale_type == 'aerosol'):
#     # rdkit과 rdkit-md의 f1-score가 performance가 동일. auc가 더 높은 rdkit-md 사용
#     args.model = 'logistic'
#     args.fp_type = 'rdkit'
#     args.add_md = True
#     args.cat3tohigh = False
#     is_smote = True
# elif (args.tg_num == 412) & (args.inhale_type == 'vapour'):
#     args.model = 'mlp'
#     args.fp_type = 'maccs'
#     args.add_md = False
#     args.cat3tohigh = False
#     is_smote = True
# elif (args.tg_num == 413) & (args.inhale_type == 'aerosol'):
#     # lda, dt, gbt 세 모델의 performance가 동일. auc가 제일 높은 dt 사용
#     args.model = 'dt'
#     args.fp_type = 'maccs'
#     args.add_md = True
#     args.cat3tohigh = False
#     is_smote = True
# elif (args.tg_num == 413) & (args.inhale_type == 'vapour'):
#     args.model = 'rf'
#     args.fp_type = 'morgan'
#     args.add_md = True
#     args.cat3tohigh = False
#     is_smote = True

# logging.info('=================================')
# logging.info('tg{} {} {}'.format(args.tg_num, args.inhale_type, args.model))
# logging.info('Fingerprints: {}, Use Descriptors: {}'.format(args.fp_type, args.add_md))

# df, drop_idx, x, y = load_data(path = '../data', args = args)
# df = df.drop(drop_idx).reset_index(drop = True)
# x_train, x_test, y_train, y_test, train_idx, test_idx = data_split(x, y, args.splitseed)


# train_smiles = df.SMILES[train_idx]
# train_ms = [Chem.MolFromSmiles(i) for i in train_smiles]
# train_maccs = [MACCSkeys.GenMACCSKeys(i) for i in train_ms]

# train_sim_mat = np.zeros((len(train_maccs), len(train_maccs)))
# for i in range(len(train_maccs)):
#     for j in range(len(train_maccs)):
#         if i <= j:
#             sim = DataStructs.TanimotoSimilarity(train_maccs[i], train_maccs[j])
#             train_sim_mat[i, j] = sim
#             train_sim_mat[j, i] = sim

# test_smiles = df.SMILES[test_idx]
# test_ms = [Chem.MolFromSmiles(i) for i in test_smiles]
# test_maccs = [MACCSkeys.GenMACCSKeys(i) for i in test_ms]

# test_sim_mat = np.zeros((len(test_maccs), len(test_maccs)))
# for i in range(len(test_maccs)):
#     for j in range(len(test_maccs)):
#         if i <= j:
#             sim = DataStructs.TanimotoSimilarity(test_maccs[i], test_maccs[j])
#             test_sim_mat[i, j] = sim
#             test_sim_mat[j, i] = sim

# train_similarities = train_sim_mat[np.triu_indices_from(train_sim_mat, k=1)]
# test_similarities = test_sim_mat[np.triu_indices_from(test_sim_mat, k=1)]


# #%%
# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.rcParams['figure.dpi'] = 300
# plt.style.use('bmh')

# plt.figure(figsize=(10, 8))
# # sns.heatmap(sim_mat)
# # plt.title(f"TG{args.tg_num}-{args.inhale_type.title()} Tanimoto Similarity Heatmap")
# sns.histplot(train_similarities, label = 'Train Similarities', kde=True, stat="density", bins=30, alpha = 0.3)
# sns.histplot(test_similarities, label = 'Test Similarities', kde=True, stat="density", bins=30, alpha = 0.3)
# plt.xlabel("Tanimoto Similarity")
# plt.ylabel("Density")
# plt.legend()
# plt.show()


# #%%
# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# train_pca = pca.fit_transform(x_train.iloc[:, :167])
# test_pca = pca.transform(x_test.iloc[:, :167])

# train_mean = np.mean(train_pca, axis=0)

# plt.figure(figsize = (12, 6))
# plt.scatter(train_pca[:, 0], train_pca[:, 1], color='blue', label='Train Set', alpha=0.5)
# plt.scatter(test_pca[:, 0], test_pca[:, 1], color='red', label='Test Set', alpha=0.5)
# plt.scatter(train_mean[0], train_mean[1], color='black', label='Train Mean', marker='x', s=100)
# plt.title("PCA Scatter Plot of Train and Test Sets")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend()
# plt.show()


# #%%
# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, random_state=42)
# train_tsne = tsne.fit_transform(x_train.iloc[:, :167])
# test_tsne = tsne.fit_transform(x_test.iloc[:, :167])

# train_mean = np.mean(train_tsne, axis=0)

# plt.figure(figsize=(12, 6))
# plt.scatter(train_tsne[:, 0], train_tsne[:, 1], color='blue', label='Train Set', alpha=0.5)
# plt.scatter(test_tsne[:, 0], test_tsne[:, 1], color='red', label='Test Set', alpha=0.5)
# plt.scatter(train_mean[0], train_mean[1], color='black', label='Train Mean', marker='x', s=100)
# plt.title("t-SNE Scatter Plot of Train and Test Sets")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.legend()
# plt.show()
# # %%
