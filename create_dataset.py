# select features
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np
import json 
import pandas as pd
import datetime
from tqdm import tqdm
import pickle
from collections import Counter
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz, issparse
from joblib import Parallel, delayed
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import argparse
import time
from sklearn.model_selection import train_test_split

save_folder = "/Users/zhengxinran/Documents/S2LAB/dataset/tif/processed_features"
dataset_folder = "/Users/zhengxinran/Documents/S2LAB/dataset/tif/combine_drebin"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def load_pickle_fast(path):
    return joblib.load(path)

def ensure_sparse(X):
    """Ensure X is a sparse matrix (CSR format)"""
    if not issparse(X):
        X = csr_matrix(X)
    elif not isinstance(X, csr_matrix):
        X = csr_matrix(X)
    return X

def process_and_save(file_path, vec, selected_feature_names, save_path):

    if not os.path.exists(file_path):
        return

    data = load_pickle_fast(file_path)
    X = data['json_features'].values
    y = np.array(data['label'])
    data['year-month'] = data['dex_date'].dt.to_period('M').astype(str)
    t = np.array(data['year-month'])
    env = np.zeros(len(y), dtype=int)

    print(f"Processing: {file_path}, Shape: {len(X)}")

    # Step 1: Transform data using the TRAINED vectorizer
    # This ensures the dimensions match what the selector expects
    X_vectorized = vec.transform(X)
    X_vectorized = ensure_sparse(X_vectorized)
    print(f"Shape after vectorization: {X_vectorized.shape}, Sparse: {issparse(X_vectorized)}")
    
    # Step 2: Select only the features that were selected for training
    # Get all feature names from the vectorizer in the EXACT same order used during training
    all_feature_names = vec.get_feature_names_out()
    
    # Create a mapping from feature names to their indices
    name_to_index = {name: idx for idx, name in enumerate(all_feature_names)}
    
    # Get indices of selected features in the SAME ORDER as training
    # This preserves feature order which is crucial for model consistency
    selected_indices = []
    for name in selected_feature_names:
        if name in name_to_index:
            selected_indices.append(name_to_index[name])
    
    if len(selected_indices) != len(selected_feature_names):
        missing = len(selected_feature_names) - len(selected_indices)
        print(f"Warning: {missing} selected features not found in test data")
    
    # Select only the features we want in the SAME ORDER as training
    X_selected = X_vectorized[:, selected_indices]
    X_selected = ensure_sparse(X_selected)
    print(f"Shape after feature selection: {X_selected.shape}, Sparse: {issparse(X_selected)}")

    # Verify dimensions match what we expect
    expected_features = len(selected_feature_names)
    actual_features = X_selected.shape[1]
    
    if actual_features != expected_features:
        print(f"WARNING: Feature count mismatch! Expected {expected_features}, got {actual_features}")
    
    # base_name = os.path.basename(file_path)
    # save_path = os.path.join(save_folder, base_name)
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save data as pickle with sparse matrix
    result = {'X': X_selected, 'y': y, 't': t, 'env': env}
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"Processed: {file_path}, Selected {X_selected.shape[1]} features, Saved to: {save_path}")


def load_training_set(sample_size=None):
    train_years = ['2014']
    train_months = [f"{m:02d}" for m in range(1, 13)]

    train_files = [f"{year}-{month}" for year in train_years for month in train_months]
    train_paths = [os.path.join(dataset_folder, f"{name}", f"features.pkl") for name in train_files]

    df_train = []
    for path in train_paths:
        if os.path.exists(path):
            data = load_pickle_fast(path)
            if sample_size and len(data) > sample_size:
                data = data.sample(n=sample_size, random_state=1)
            df_train.append(data)
    
    df_train = pd.concat(df_train)
    df_train['year-month'] = df_train['dex_date'].dt.to_period('M').astype(str)
    print(f"Total training samples: {df_train.shape[0]}")

    return df_train


def get_feature_selector(method, X_train, y_train, n_features = 10000):

    X_train = ensure_sparse(X_train)

    start_time = time.time()
    if method == 'linearsvc':
        model = LinearSVC(dual=False, max_iter=10000, random_state=1, penalty='l2')
        model.fit(X_train, y_train)
        selector = SelectFromModel(model, max_features=n_features, prefit=True)
        end_time = time.time()
    
    elif method == 'logistic':
        model = LogisticRegression(max_iter=10000, random_state=1, penalty='l2', solver='saga')
        model.fit(X_train, y_train)
        selector = SelectFromModel(model, max_features=n_features, prefit=True)
        end_time = time.time()
    
    elif method == 'randomforest':
        # Note: RandomForest might be slow with sparse matrices and high dimensions
        # Use SGDClassifier with log loss as alternative
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
        model.fit(X_train, y_train)
        selector = SelectFromModel(model, max_features=n_features, prefit=True)
        end_time = time.time()
    
    elif method == 'chi2':
        selector = SelectKBest(chi2, k=n_features)
        selector.fit(X_train, y_train)
        end_time = time.time()
    
    elif method == 'variance':
        selector = VarianceThreshold(threshold=0.003)
        selector.fit(X_train)
        end_time = time.time()
    
    elif method == 'none':
        selector = None
        end_time = time.time()
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
        end_time = time.time()
    
    print(f"Feature selection time: {end_time - start_time:.2f}s")
    
    return selector

def generate_selector(method='randomforest', n_features=10000, batch_size=10000):
    df_train = load_training_set()

    print(f"Total training samples: {df_train.shape[0]}")

    X_train = df_train['json_features'].values
    y_train = df_train['label'].values
    t_train = df_train['year-month'].values

    # Create output directory
    os.makedirs(save_folder, exist_ok=True)

    # Step 1: First, fit the vectorizer ONCE on all training data
    print("Fitting vectorizer on all training data...")
    vec = DictVectorizer(sparse=True, sort=True)
    
    # Process in batches for large datasets to save memory
    if len(X_train) > batch_size:
        print(f"Processing in batches of {batch_size}...")
        X_batches = []
        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            batch = X_train[i:end]
            print(f"  Processing batch {i}-{end} of {len(X_train)}...")
            # Only fit on the first batch, transform the rest
            if i == 0:
                X_batch = vec.fit_transform(batch)
            else:
                X_batch = vec.transform(batch)
            X_batch = ensure_sparse(X_batch)
            X_batches.append(X_batch)
        X_train_vectorized = vstack(X_batches)
    else:
        X_train_vectorized = vec.fit_transform(X_train)
        X_train_vectorized = ensure_sparse(X_train_vectorized)
    
    # Get all feature names in the EXACT order used by the vectorizer
    all_feature_names = vec.get_feature_names_out()
    print(f"After vectorization - Shape: {X_train_vectorized.shape}, Features: {len(all_feature_names)}")
    print(f"Is sparse: {issparse(X_train_vectorized)}")

    # IMPORTANT: Save the vectorizer BEFORE applying any feature selection
    # We need to apply the SAME vectorization to test data
    joblib.dump(vec, os.path.join(save_folder, f'vectorizer_{method}.pkl'))

    selector_path = os.path.join(save_folder, f'selector_{method}.pkl')
    selected_features_path = os.path.join(save_folder, f'selected_features_{method}.txt')
    
    # Step 2: Apply feature selection on the VECTORIZED data
    print(f"Applying feature selection using {method}...")
    
    if method == 'none':
        # No feature selection - keep all features
        selected_feature_names = all_feature_names
        X_train_selected = X_train_vectorized
    elif os.path.exists(selector_path) and os.path.exists(selected_features_path):
        print(f"Loading existing selector and selected features for method '{method}'...")
        selector = joblib.load(selector_path)
        with open(selected_features_path, 'r') as f:
            selected_feature_names = [line.strip() for line in f.readlines()]
        name_to_index = {name: idx for idx, name in enumerate(all_feature_names)}
        selected_indices = [name_to_index[name] for name in selected_feature_names if name in name_to_index]
        X_train_selected = X_train_vectorized[:, selected_indices]
        X_train_selected = ensure_sparse(X_train_selected)
    else:
        print(f"Applying feature selection using {method}...")
        selector = get_feature_selector(method, X_train_vectorized, y_train, n_features)

        if hasattr(selector, 'get_support'):
            # Get a boolean mask of selected features
            feature_mask = selector.get_support()
            
            # Get the names of selected features IN THE EXACT SAME ORDER
            selected_feature_names = all_feature_names[feature_mask]
            # Save the selected feature names for consistent selection
            print(f"Selected {len(selected_feature_names)} features")
            with open(os.path.join(save_folder, f'selected_features.txt'), 'w') as f:
                for feature in selected_feature_names:
                    f.write(f"{feature}\n")
            
            # Show feature importances if available
            if hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'coef_'):
                feature_importances = np.abs(selector.estimator_.coef_[0])
                sorted_indices = np.argsort(-feature_importances)
                top_features = all_feature_names[sorted_indices[:10]]
                top_importances = feature_importances[sorted_indices[:10]]

                print("Top 10 feature importances:")
                for name, importance in zip(top_features, top_importances):
                    print(f"{name}: {importance}")
                
                # Save feature importances
                feature_importance_dict = {
                    name: float(importance) 
                    for name, importance in zip(all_feature_names, feature_importances)
                }
                with open(os.path.join(save_folder, f'feature_importances_{method}.json'), 'w') as f:
                    json.dump(feature_importance_dict, f)
            
            # Save selector for reference
            joblib.dump(selector, os.path.join(save_folder, f'selector_{method}.pkl'))
            
            # Create a mapping from feature names to their indices
            name_to_index = {name: idx for idx, name in enumerate(all_feature_names)}
            
            # Get indices of selected features
            selected_indices = [name_to_index[name] for name in selected_feature_names]
            
            # Select features from training data in the EXACT SAME ORDER
            X_train_selected = X_train_vectorized[:, selected_indices]
            X_train_selected = ensure_sparse(X_train_selected)
        else:
            # Fallback for custom selectors that don't have get_support()
            X_train_selected = selector.transform(X_train_vectorized)
            # In this case, we use the first n_features names but the ACTUAL order depends on the selector
            selected_feature_names = all_feature_names[:n_features]
    
    # Save the selected feature names for consistent selection
    print(f"Selected {len(selected_feature_names)} features")
    with open(os.path.join(save_folder, f'selected_features_{method}.txt'), 'w') as f:
        for feature in selected_feature_names:
            f.write(f"{feature}\n")
    
    print(f"Selected {len(selected_feature_names)} features")
    print(f"Final matrix shape: {X_train_selected.shape}")

    year_month_train = np.array([datetime.strptime(t, "%Y-%m") for t in t_train])
    envs_train = (np.array([date.month for date in year_month_train]) - 1) // 3
    
    X_train_split, X_val_split, y_train_split, y_val_split, envs_train_split, envs_val_split, t_train_split, t_val_split = train_test_split(
        X_train_selected, y_train, envs_train, t_train, test_size=0.2, random_state=42, stratify=y_train
    )

    with open(os.path.join(save_folder, f'train_data.pkl'), 'wb') as f:
        pickle.dump({'X': X_train_split, 'y': y_train_split, 't': t_train_split, 'env': envs_train_split}, f)
    with open(os.path.join(save_folder, f'val_data.pkl'), 'wb') as f:
        pickle.dump({'X': X_val_split, 'y': y_val_split, 't': t_val_split, 'env': envs_val_split}, f)

    months = ['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
                '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12',
                '2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
                '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12',
                '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
                '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12',
                '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
                '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
                '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']
    
    for month in months:
        print(f"Processing {month}")
        test_paths = os.path.join(dataset_folder, f"{month}", "features.pkl")
        save_path = os.path.join(save_folder, f"{month}.pkl")
        process_and_save(test_paths, vec, selected_feature_names, save_path)

    # check
    # load train data
    with open(os.path.join(save_folder, f'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    print(f"train data shape: {train_data['X'].shape}, {train_data['y'].shape}, {train_data['env'].shape}, {train_data['t'].shape}")
    # load val data
    with open(os.path.join(save_folder, f'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    print(f"val data shape: {val_data['X'].shape}, {val_data['y'].shape}, {val_data['env'].shape}, {val_data['t'].shape}")
    # load test data
    with open(os.path.join(save_folder, f'2021-03.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    print(f"test data shape: {test_data['X'].shape}, {test_data['y'].shape}, {test_data['env'].shape}, {test_data['t'].shape}")


if __name__ == '__main__':
    generate_selector()
    # with open(os.path.join("/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features", f'2021-01.pkl'), 'rb') as f:
    #     test_data = pickle.load(f)
    # print(f"test data shape: {test_data['X'].shape}, {test_data['y'].shape}, {test_data['env'].shape}, {test_data['t'].shape}")
    # load train data
    # with open(os.path.join(save_folder, f'train_data.pkl'), 'rb') as f:
    #     train_data = pickle.load(f)
    # print(f"train data shape: {train_data['X'].shape}, {train_data['y'].shape}, {train_data['env'].shape}, {train_data['t'].shape}")
    # print(f"distribution of train data: {Counter(train_data['y']), Counter(train_data['env'])}")
    # # load val data
    # with open(os.path.join(save_folder, f'val_data.pkl'), 'rb') as f:
    #     val_data = pickle.load(f)
    # print(f"val data shape: {val_data['X'].shape}, {val_data['y'].shape}, {val_data['env'].shape}, {val_data['t'].shape}")
    # print(f"distribution of val data: {Counter(val_data['y']), Counter(val_data['env'])}")
    # # load test data
    # with open(os.path.join(save_folder, f'2021-03.pkl'), 'rb') as f:
    #     test_data = pickle.load(f)
    # print(f"test data shape: {test_data['X'].shape}, {test_data['y'].shape}, {test_data['env'].shape}, {test_data['t'].shape}")
    # print(f"distribution of test data: {Counter(test_data['y']), Counter(test_data['env'])}")



    