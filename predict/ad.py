import numpy as np


#4-2. DoA 함수 생성
# 마할라노비스 거리 계산 함수
def mahalanobis_distance(fingerprint, train_fingerprints, epsilon=1e-10):
    centroid = np.mean(train_fingerprints, axis=0)
    cov_matrix = np.cov(train_fingerprints.T) + np.eye(train_fingerprints.shape[1]) * epsilon
    return distance.mahalanobis(fingerprint, centroid, np.linalg.inv(cov_matrix))


# 유클리드 거리, threshold 계산 함수
def euclidean_distance(fingerprint, train_fingerprints):
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

#4-3. 최적 모델을 활용하여 화학물질 독성 예측하는 함수 생성
def predict_from_csv(model_file, selector_file, scaler_file, best_params_file, training_data_file_path, target_data_file_path, output_csv, fingerprint_type, k=3, Z=0.5):
    # Load model, feature selector, and scaler
    print("Loading model, selector, and scaler...")
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    selector = joblib.load(selector_file)

    
    # Load best hyperparameters if the file exists
    if os.path.exists(best_params_file):
        print("Loading best hyperparameters...")
        best_params = joblib.load(best_params_file)
        # Update the model with the best hyperparameters
        model.set_params(**best_params)
    else:
        print("Best hyperparameters file not found. Proceeding with default parameters.")
    
    # Process training data
    print("Processing training data...")
    train_data = load_and_preprocess_training_data(training_data_file_path)
    train_data = train_data[['SMILES', 'Mol']]
    train_data['Fingerprint'] = train_data['Mol'].apply(lambda mol: generate_fingerprint(mol, fingerprint_type) if mol is not None else None)
    train_data = train_data[train_data['Fingerprint'].notnull()]
    train_fingerprints = np.array(list(train_data['Fingerprint'])).reshape(train_data.shape[0], -1)
    print(f"Number of training fingerprints: {train_fingerprints.shape[0]}")
    print(f"Training fingerprints shape: {train_fingerprints.shape}")

    # Process target data
    print("Processing target data...")
    target_data = load_and_preprocess_target_data(target_data_file_path)
    target_data = target_data[['SMILES', 'Mol']]
    target_data['Fingerprint'] = target_data['Mol'].apply(lambda mol: generate_fingerprint(mol, fingerprint_type) if mol is not None else None)
    target_data = target_data[target_data['Fingerprint'].notnull()]
    target_fingerprints = np.array(list(target_data['Fingerprint'])).reshape(target_data.shape[0], -1)
    print(f"Number of target fingerprints: {target_fingerprints.shape[0]}")
    print(f"Target fingerprints shape: {target_fingerprints.shape}")

    # Scale and select features using the loaded scaler
    print("Scaling and selecting features...")
    train_fingerprints_scaled = scaler.transform(train_fingerprints)
    train_fingerprints_selected = selector.transform(train_fingerprints_scaled)
    target_fingerprints_scaled = scaler.transform(target_fingerprints)
    target_fingerprints_selected = selector.transform(target_fingerprints_scaled)
    print(f"Scaled and selected training fingerprints shape: {train_fingerprints_selected.shape}")
    print(f"Scaled and selected target fingerprints shape: {target_fingerprints_selected.shape}")

    # Calculate Applicability Domain threshold
    print("Calculating Applicability Domain threshold...")
    D_T = calculate_ad_threshold(train_fingerprints_selected, k, Z)
    print(f"Applicability Domain threshold (D_T): {D_T}")

    # Predict target data
    print("Predicting target data...")
    predictions = []
    #reliability_mahalanobis = []
    reliability_euclidean = []
    mahalanobis_distances = []
    euclidean_distances = []
    for fp in target_fingerprints_selected:
        md = mahalanobis_distance(fp, train_fingerprints_selected)
        ed = euclidean_distance(fp, train_fingerprints_selected)
        mahalanobis_distances.append(md)
        euclidean_distances.append(ed)
        
        #if md < threshold:
        #    reliability_mahalanobis.append("reliable")
        #else:
        #    reliability_mahalanobis.append("unreliable")
        
        if ed < D_T:
            reliability_euclidean.append("reliable")
        else:
            reliability_euclidean.append("unreliable")
        
        pred = model.predict(fp.reshape(1, -1))
        predictions.append(pred[0])


#3-3. 모델 평가 및 저장 함수

def evaluate_models(X_train, y_train, X_test, y_test, fingerprint_type, model_params, scoring, model_save_dir, apply_scaling=True):
    """Train, evaluate, and save machine learning models using different fingerprint types and model parameters."""
    results = {}
    best_estimators = {}
    test_predictions = pd.DataFrame(index=range(len(X_test)))
    reliability_dict = {}
    
    for name, mp in tqdm(model_params.items(), desc="Model"):
        print(f"Training {name} with {fingerprint_type} fingerprint...")
        
        # Feature selection
        X_train_selected, selector, selected_features = feature_selection(X_train, y_train)
        
        # Save feature selector and selected feature indices
        selector_file = os.path.join(model_save_dir, f'{fingerprint_type}_{name}_selector.pkl')
        selected_features_file = os.path.join(model_save_dir, f'{fingerprint_type}_{name}_selected_features.pkl')
        joblib.dump(selector, selector_file)
        joblib.dump(selected_features, selected_features_file)
        
        # Stratified K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=5)
        
        # Create a pipeline with SMOTE and the model
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', mp['model'])
        ])
        
        # Perform Grid Search
        grid = GridSearchCV(estimator=pipeline, param_grid=mp['params'], cv=skf, scoring='f1')
        grid.fit(X_train_selected, y_train)
        
        best_estimators[name] = grid.best_estimator_
        
        # Save model
        model_save_path = os.path.join(model_save_dir, f'{fingerprint_type}_{name}.pkl')
        joblib.dump(grid.best_estimator_, model_save_path)
        
        # Save best hyperparameters
        best_params_path = os.path.join(model_save_dir, f'{fingerprint_type}_{name}_best_params.pkl')
        joblib.dump(grid.best_params_, best_params_path)
        
        # Load the saved scaler, if it exists
        scaler_file = os.path.join(model_save_dir, f'{fingerprint_type}_scaler.pkl')
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
        else:
            scaler = None
        
        # Collect results
        results[name] = {}
        
        for metric_name, metric in tqdm(scoring.items(), desc=f"Evaluating {name}"):
            scores = cross_val_score(grid.best_estimator_, X_train_selected, y_train, cv=skf, scoring=metric)
            results[name][metric_name] = scores.mean()

            # Scale and select features for the test set
            if apply_scaling and scaler is not None:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            X_test_selected = selector.transform(X_test_scaled)
            test_pred = grid.best_estimator_.predict(X_test_selected)
            
            # Calculate Applicability Domain threshold
            D_T = calculate_ad_threshold(X_train_selected)
            reliability = [euclidean_distance(fp, X_train_selected) < D_T for fp in X_test_selected]
            
            reliability_dict[f'{name}_reliability'] = reliability
            
            # Directly calculate metric for the test set
            if metric_name == 'accuracy':
                test_score = accuracy_score(y_test, test_pred)
            elif metric_name == 'f1':
                test_score = f1_score(y_test, test_pred)
            elif metric_name == 'roc_auc':
                test_score = roc_auc_score(y_test, test_pred)
            elif metric_name == 'precision':
                test_score = precision_score(y_test, test_pred, zero_division=1)
            elif metric_name == 'recall':
                test_score = recall_score(y_test, test_pred, zero_division=1)
            else:
                test_score = metric(y_test, test_pred)
            
            results[name][f'test_{metric_name}'] = test_score
        
        test_predictions[name] = test_pred
    
    results_df = pd.DataFrame(results).transpose()
    reliability_df = pd.DataFrame(reliability_dict)
    
    # Save results
    results_path = os.path.join(model_save_dir, f'results_{fingerprint_type}.csv')
    results_df.to_csv(results_path)
    
    # Save test predictions
    test_predictions_path = os.path.join(model_save_dir, f'test_predictions_{fingerprint_type}.csv')
    test_predictions.to_csv(test_predictions_path, index=False)
    
    # Save reliability results
    reliability_path = os.path.join(model_save_dir, f'reliability_{fingerprint_type}.csv')
    reliability_df.to_csv(reliability_path, index=False)
    
    print(f"Loaded results for {fingerprint_type}:")
    print(results_df)
    
    return results_df, best_estimators, test_predictions, reliability_df