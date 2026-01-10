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
import time
from sklearn.model_selection import train_test_split
import argparse

save_folder = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features_month"
dataset_folder = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/drebin_new"
family_dict_path = '/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/combine_drebin/family_dict.json'

os.makedirs(save_folder, exist_ok=True)

def load_pickle_fast(path):
    return joblib.load(path)

def ensure_sparse(X):
    """Ensure X is a sparse matrix (CSR format)"""
    if not issparse(X):
        X = csr_matrix(X)
    elif not isinstance(X, csr_matrix):
        X = csr_matrix(X)
    return X

def compute_env_ids(t_train, env_split_mode='quarter', n_envs=4):
    """
    Compute environment IDs based on time stamps.
    
    Args:
        t_train: Array of time strings in format "YYYY-MM"
        env_split_mode: 'quarter' for quarterly split, 'month' for monthly split, or 'uniform' for uniform split
        n_envs: Number of environments for uniform split (only used when env_split_mode='uniform')
    
    Returns:
        Array of environment IDs
    """
    year_month_train = np.array([datetime.strptime(t, "%Y-%m") for t in t_train])
    
    if env_split_mode == 'quarter':
        # Quarterly split: Q1 (0), Q2 (1), Q3 (2), Q4 (3)
        envs = (np.array([date.month for date in year_month_train]) - 1) // 3
    elif env_split_mode == 'month':
        # Monthly split: Jan (0), Feb (1), ..., Dec (11)
        envs = np.array([date.month for date in year_month_train]) - 1
    elif env_split_mode == 'uniform':
        # Uniform split: evenly divide samples into n_envs environments
        n_samples = len(t_train)
        # Assign each sample to an environment based on its index
        envs = np.array([i % n_envs for i in range(n_samples)])
    else:
        raise ValueError(f"Unknown env_split_mode: {env_split_mode}. Use 'quarter', 'month', or 'uniform'")
    
    return envs

def load_data_from_file(file_path):
    """Load and parse data from pickle file, handling different formats"""
    if not os.path.exists(file_path):
        return None
    
    data = load_pickle_fast(file_path)
    
    # Handle different data formats
    if isinstance(data, dict) and 'json_features' in data:
        # New format: dictionary with arrays
        json_features = data.get('json_features', [])
        label = data.get('label', [])
        family = data.get('family', [])
        vt_scan_date = data.get('vt_scan_date', [])
        
        # Convert to numpy arrays if needed
        if isinstance(json_features, np.ndarray):
            X = json_features.tolist()
        else:
            X = list(json_features)
        
        y = np.array(label)
        y_family = np.array(family)
        
        # Use vt_scan_date for time grouping
        if isinstance(vt_scan_date, np.ndarray):
            if len(vt_scan_date) > 0 and isinstance(vt_scan_date[0], (pd.Timestamp, np.datetime64)):
                vt_scan_dates = pd.to_datetime(vt_scan_date)
            else:
                vt_scan_dates = pd.to_datetime(vt_scan_date)
        else:
            vt_scan_dates = pd.to_datetime(vt_scan_date)
        
        # Create time period strings
        t = np.array([pd.to_datetime(d).to_period('M').strftime('%Y-%m') if pd.notna(d) else 'unknown' for d in vt_scan_dates])
        env = np.zeros(len(y), dtype=int)
        
    elif isinstance(data, pd.DataFrame):
        # Old format: DataFrame
        X = data['json_features'].values.tolist()
        y = np.array(data['label'])
        y_family = np.array(data['family'])
        
        # Try to use vt_scan_date if available, otherwise use dex_date
        if 'vt_scan_date' in data.columns:
            vt_scan_dates = pd.to_datetime(data['vt_scan_date'])
            t = np.array([pd.to_datetime(d).to_period('M').strftime('%Y-%m') if pd.notna(d) else 'unknown' for d in vt_scan_dates])
        else:
            data['year-month'] = data['date'].dt.to_period('M').astype(str)
            t = np.array(data['year-month'])
        
        env = np.zeros(len(y), dtype=int)
    else:
        print(f"Error: Unknown data format in {file_path}")
        return None
    
    return {'X': X, 'y': y, 'y_family': y_family, 't': t, 'env': env}

def process_single_file(file_path, vec, selected_feature_names, save_path):
    """Process a single file using existing vectorizer and selected features"""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return False

    # Load data
    data_dict = load_data_from_file(file_path)
    if data_dict is None:
        return False
    
    X = data_dict['X']
    y = data_dict['y']
    y_family = data_dict['y_family']
    t = data_dict['t']
    env = data_dict['env']
    
    
    if os.path.exists(family_dict_path):
        with open(family_dict_path, 'r') as f:
            family_dict = json.load(f)
        y_family_encoded = np.array([family_dict.get(fam, family_dict.get('unknown', 0)) for fam in y_family])
    else:
        print("Warning: family_dict.json not found, using family names as-is")
        y_family_encoded = y_family
    
    print(f"Processing: {file_path}, Shape: {len(X)}")
    
    # Step 1: Transform data using the TRAINED vectorizer
    X_vectorized = vec.transform(X)
    X_vectorized = ensure_sparse(X_vectorized)
    print(f"Shape after vectorization: {X_vectorized.shape}, Sparse: {issparse(X_vectorized)}")
    
    # Step 2: Select only the features that were selected for training
    all_feature_names = vec.get_feature_names_out()
    name_to_index = {name: idx for idx, name in enumerate(all_feature_names)}
    
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
    
    # Save data as pickle with sparse matrix
    result = {'X': X_selected, 'y': y, 't': t, 'env': env, 'y_family': y_family_encoded}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Processed: {file_path}, Selected {X_selected.shape[1]} features, Saved to: {save_path}")
    return True


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
    df_train['year-month'] = df_train['date'].dt.to_period('M').astype(str)
    print(f"Total training samples: {df_train.shape[0]}")

    return df_train


def get_feature_selector(method, X_train, y_train, n_features=10000):

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
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42,n_jobs=-1)
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

def process_dataset(mode='regenerate', method='randomforest', n_features=10000, batch_size=10000, 
                   feature_list_file=None, vectorizer_file=None, input_files=None, output_folder=None,
                   predefined_features_file=None, process_train_val=True, test_list=None, env_split_mode='quarter', n_envs=4):
    """
    Unified function to process dataset with multiple modes:
    
    Mode 1 (mode='process'): Process dataset using existing feature list
        - Loads vectorizer and feature list from files
        - Processes input files using existing features
        
    Mode 2 (mode='regenerate'): Extract features using different selector, then process test sets
        - Uses existing vectorizer (or creates new one)
        - Trains a NEW selector with specified method
        - Optionally processes train/val sets (if process_train_val=True)
        - Processes test sets (all or specified months via test_list)
        
    Mode 3 (mode='predefine'): Use external feature list, process train/val/test sets
        - Loads predefined feature list from external file
        - Uses existing vectorizer (or creates new one)
        - Processes train, val, and test sets (all or specified months via test_list) with predefined features
        
    Args:
        mode: 'process', 'regenerate', or 'predefine'
        method: Feature selection method (for regenerate mode)
        n_features: Number of features to select (for regenerate mode)
        batch_size: Batch size for processing large datasets
        feature_list_file: Path to feature list txt file (for process mode)
        vectorizer_file: Path to vectorizer pkl file (for process/predefine/regenerate modes)
        input_files: List of input file paths (for process mode)
        output_folder: Output folder for processed files
        predefined_features_file: Path to external feature list file (for predefine mode, required)
        process_train_val: Whether to process train/val sets in regenerate mode (default: True)
        test_list: List of test months to process (e.g., ['2015-01', '2015-02']) or None for all months
    """
    if mode == 'process':
        if feature_list_file is None or vectorizer_file is None:
            raise ValueError("feature_list_file and vectorizer_file are required for process mode")
        if input_files is None:
            raise ValueError("input_files is required for process mode")
        return _process_with_existing_features(feature_list_file, vectorizer_file, input_files, output_folder)
    elif mode == 'regenerate':
        return _process_regenerate_mode(method, n_features, batch_size, env_split_mode, n_envs, vectorizer_file, output_folder, process_train_val, test_list)
    elif mode == 'predefine':
        if predefined_features_file is None:
            raise ValueError("predefined_features_file is required for predefine mode")
        return _process_predefine_mode(predefined_features_file, vectorizer_file, output_folder, batch_size, test_list, env_split_mode)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'process', 'regenerate', or 'predefine'")


def _process_with_existing_features(feature_list_file, vectorizer_file, input_files, output_folder=None):
    """Mode: Process dataset using existing feature list"""
    if output_folder is None:
        output_folder = save_folder
    
    # Load selected features
    if not os.path.exists(feature_list_file):
        raise FileNotFoundError(f"Feature list file not found: {feature_list_file}")
    
    print(f"Loading selected features from {feature_list_file}...")
    with open(feature_list_file, 'r') as f:
        selected_feature_names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(selected_feature_names)} selected features")
    
    # Load vectorizer
    if not os.path.exists(vectorizer_file):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_file}")
    
    print(f"Loading vectorizer from {vectorizer_file}...")
    vec = joblib.load(vectorizer_file)
    print("Vectorizer loaded successfully")
    
    # Process input files
    if isinstance(input_files, dict):
        # Dictionary mapping input -> output paths
        for input_path, output_path in input_files.items():
            if output_folder:
                output_path = os.path.join(output_folder, os.path.basename(output_path))
            print(f"\nProcessing {input_path}")
            process_single_file(input_path, vec, selected_feature_names, output_path)
    elif isinstance(input_files, list):
        # List of input file paths
        for input_path in input_files:
            # Extract month name from path (e.g., /path/to/2024-01/features.pkl -> 2024-01)
            # Path format: .../YYYY-MM/features.pkl
            path_parts = input_path.split(os.sep)
            month_name = None
            for part in path_parts:
                # Check if part matches YYYY-MM format
                if len(part) == 7 and part[4] == '-' and part[:4].isdigit() and part[5:].isdigit():
                    month_name = part
                    break
            
            if month_name:
                output_path = os.path.join(output_folder, f"{month_name}.pkl")
            else:
                # Fallback: use basename without extension
                base_name = os.path.basename(input_path).replace('.pkl', '')
                output_path = os.path.join(output_folder, f"{base_name}.pkl")
            
            print(f"\nProcessing {input_path}")
            process_single_file(input_path, vec, selected_feature_names, output_path)
    else:
        raise ValueError("input_files must be a list or dict")
    
    return vec, selected_feature_names

def _process_regenerate_mode(method='randomforest', n_features=10000, batch_size=10000, env_split_mode='quarter', n_envs=4,
                             vectorizer_file=None, output_folder=None, 
                             process_train_val=True, test_list=None):
    """
    Regenerate mode: Extract features using different selector, then process train/val/test sets.
    Uses existing vectorizer (or creates new one), trains a NEW selector, processes train/val/test sets.
    
    Args:
        method: Feature selection method
        n_features: Number of features to select
        batch_size: Batch size for processing
        vectorizer_file: Path to existing vectorizer (optional, creates new if not provided)
        output_folder: Output folder for processed files
        process_train_val: Whether to process and save train/val sets (default: True)
        test_list: List of test months to process (None for all months)
    """
    if output_folder is None:
        output_folder = save_folder
    
    # Load or create vectorizer
    if vectorizer_file and os.path.exists(vectorizer_file):
        print(f"Loading existing vectorizer from {vectorizer_file}...")
        vec = joblib.load(vectorizer_file)
        print("Vectorizer loaded successfully")
        
        # Load training data to train new selector
        df_train = load_training_set()
        X_train = df_train['json_features'].values
        y_train = df_train['label'].values
        
        # Vectorize training data
        print("Vectorizing training data...")
        if len(X_train) > batch_size:
            print(f"Processing in batches of {batch_size}...")
            X_batches = []
            for i in range(0, len(X_train), batch_size):
                end = min(i + batch_size, len(X_train))
                batch = X_train[i:end]
                X_batch = vec.transform(batch)
                X_batch = ensure_sparse(X_batch)
                X_batches.append(X_batch)
            X_train_vectorized = vstack(X_batches)
        else:
            X_train_vectorized = vec.transform(X_train)
            X_train_vectorized = ensure_sparse(X_train_vectorized)
    else:
        # Create new vectorizer (same as train mode)
        print("Creating new vectorizer...")
        df_train = load_training_set()
        X_train = df_train['json_features'].values
        y_train = df_train['label'].values
        
        vec = DictVectorizer(sparse=True, sort=True)
        if len(X_train) > batch_size:
            print(f"Processing in batches of {batch_size}...")
            X_batches = []
            for i in range(0, len(X_train), batch_size):
                end = min(i + batch_size, len(X_train))
                batch = X_train[i:end]
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
        
        # Save vectorizer
        vec_path = os.path.join(output_folder, f'vectorizer_{method}_regenerated.pkl')
        joblib.dump(vec, vec_path)
        print(f"Saved vectorizer to: {vec_path}")
    
    all_feature_names = vec.get_feature_names_out()
    print(f"Vectorized training data - Shape: {X_train_vectorized.shape}, Features: {len(all_feature_names)}")
    
    # Train NEW selector with specified method
    print(f"Training NEW selector using method: {method}...")
    
    selector = get_feature_selector(method, X_train_vectorized, y_train, n_features)
    
    # Get selected features
    if hasattr(selector, 'get_support'):
        feature_mask = selector.get_support()
        selected_feature_names = all_feature_names[feature_mask]
    else:
        X_train_selected = selector.transform(X_train_vectorized)
        selected_feature_names = all_feature_names[:n_features]
    
    print(f"Selected {len(selected_feature_names)} features using {method}")
    
    # Save new selector and features
    selector_path = os.path.join(output_folder, f'selector_{method}_regenerated.pkl')
    features_path = os.path.join(output_folder, f'selected_features_{method}_regenerated.txt')
    
    joblib.dump(selector, selector_path)
    with open(features_path, 'w') as f:
        for feature in selected_feature_names:
            f.write(f"{feature}\n")
    
    print(f"Saved selector to: {selector_path}")
    print(f"Saved features to: {features_path}")
    
    # Process train/val sets if requested
    if process_train_val:
        print("\nProcessing training and validation sets...")
        y_family = df_train['family'].values
        t_train = df_train['year-month'].values
        
        with open(family_dict_path, 'r') as f:
            family_dict = json.load(f)
        y_family_encoded = np.array([family_dict[family] for family in y_family])
        
        # Select features from training data
        all_feature_names = vec.get_feature_names_out()
        name_to_index = {name: idx for idx, name in enumerate(all_feature_names)}
        selected_indices = [name_to_index[name] for name in selected_feature_names if name in name_to_index]
        X_train_selected = X_train_vectorized[:, selected_indices]
        X_train_selected = ensure_sparse(X_train_selected)
        
        # Split into train/val
        envs_train = compute_env_ids(t_train, env_split_mode=env_split_mode, n_envs=n_envs)
        
        X_train_split, X_val_split, y_train_split, y_val_split, envs_train_split, envs_val_split, t_train_split, t_val_split, y_family_train_split, y_family_val_split = train_test_split(
            X_train_selected, y_train, envs_train, t_train, y_family_encoded, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Save train/val data
        with open(os.path.join(output_folder, 'train_data.pkl'), 'wb') as f:
            pickle.dump({'X': X_train_split, 'y': y_train_split, 't': t_train_split, 'env': envs_train_split, 'y_family': y_family_train_split}, f)
        with open(os.path.join(output_folder, 'val_data.pkl'), 'wb') as f:
            pickle.dump({'X': X_val_split, 'y': y_val_split, 't': t_val_split, 'env': envs_val_split, 'y_family': y_family_val_split}, f)
        
        print("Saved train_data.pkl and val_data.pkl")
    
    # Process test sets
    if test_list is None:
        print("\nProcessing all test sets...")
        months = _get_all_test_months()
    else:
        print(f"\nProcessing {len(test_list)} specified test months...")
        months = test_list
    
    for month in months:
        input_path = os.path.join(dataset_folder, f"{month}", "features.pkl")
        output_path = os.path.join(output_folder, f"{month}.pkl")
        if os.path.exists(input_path):
            print(f"Processing {month}...")
            process_single_file(input_path, vec, selected_feature_names, output_path)
        else:
            print(f"  Warning: Input file not found: {input_path}, skipping...")
    
    return vec, selected_feature_names

def _process_predefine_mode(predefined_features_file, vectorizer_file=None, output_folder=None, batch_size=10000, test_list=None, env_split_mode='quarter', n_envs=4):
    """
    Predefine mode: Use external feature list, process train/val/test sets.
    Loads predefined feature list from external file, uses existing vectorizer (or creates new one),
    processes train, val, and test sets (all or specified months) with predefined features.
    
    Args:
        predefined_features_file: Path to external feature list file
        vectorizer_file: Path to existing vectorizer (optional, creates new if not provided)
        output_folder: Output folder for processed files
        batch_size: Batch size for processing
        test_list: List of test months to process (None for all months)
    """
    if output_folder is None:
        output_folder = save_folder
    
    # Load predefined features
    if not os.path.exists(predefined_features_file):
        raise FileNotFoundError(f"Predefined features file not found: {predefined_features_file}")
    
    print(f"Loading predefined features from {predefined_features_file}...")
    with open(predefined_features_file, 'r') as f:
        selected_feature_names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(selected_feature_names)} predefined features")
    
    # Load or create vectorizer
    if vectorizer_file and os.path.exists(vectorizer_file):
        print(f"Loading existing vectorizer from {vectorizer_file}...")
        vec = joblib.load(vectorizer_file)
        print("Vectorizer loaded successfully")
    else:
        # Create new vectorizer from training data
        print("Creating new vectorizer from training data...")
        df_train = load_training_set()
        X_train = df_train['json_features'].values
        
        vec = DictVectorizer(sparse=True, sort=True)
        if len(X_train) > batch_size:
            print(f"Processing in batches of {batch_size}...")
            X_batches = []
            for i in range(0, len(X_train), batch_size):
                end = min(i + batch_size, len(X_train))
                batch = X_train[i:end]
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
        
        # Save vectorizer
        vec_path = os.path.join(output_folder, 'vectorizer_predefined.pkl')
        joblib.dump(vec, vec_path)
        print(f"Saved vectorizer to: {vec_path}")
    
    # Verify all predefined features exist in vectorizer
    all_feature_names = vec.get_feature_names_out()
    name_to_index = {name: idx for idx, name in enumerate(all_feature_names)}
    
    missing_features = [name for name in selected_feature_names if name not in name_to_index]
    if missing_features:
        print(f"\n{'='*80}")
        print(f"Warning: {len(missing_features)} predefined features not found in vectorizer!")
    else:
        print(f"All {len(selected_feature_names)} predefined features found in vectorizer")
    
    # Process training and validation sets
    print("\nProcessing training and validation sets...")
    df_train = load_training_set()
    X_train = df_train['json_features'].values
    y_train = df_train['label'].values
    y_family = df_train['family'].values
    t_train = df_train['year-month'].values
    
    with open(family_dict_path, 'r') as f:
        family_dict = json.load(f)
    y_family_encoded = np.array([family_dict[family] for family in y_family])
    
    # Vectorize training data
    print("Vectorizing training data...")
    if len(X_train) > batch_size:
        X_batches = []
        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            batch = X_train[i:end]
            X_batch = vec.transform(batch)
            X_batch = ensure_sparse(X_batch)
            X_batches.append(X_batch)
        X_train_vectorized = vstack(X_batches)
    else:
        X_train_vectorized = vec.transform(X_train)
        X_train_vectorized = ensure_sparse(X_train_vectorized)
    
    # Select predefined features
    selected_indices = [name_to_index[name] for name in selected_feature_names]
    X_train_selected = X_train_vectorized[:, selected_indices]
    X_train_selected = ensure_sparse(X_train_selected)
    
    print(f"Selected features shape: {X_train_selected.shape}")
    
    # Split into train/val
    envs_train = compute_env_ids(t_train, env_split_mode=env_split_mode, n_envs=n_envs)
    
    X_train_split, X_val_split, y_train_split, y_val_split, envs_train_split, envs_val_split, t_train_split, t_val_split, y_family_train_split, y_family_val_split = train_test_split(
        X_train_selected, y_train, envs_train, t_train, y_family_encoded, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Save train/val data
    with open(os.path.join(output_folder, 'train_data.pkl'), 'wb') as f:
        pickle.dump({'X': X_train_split, 'y': y_train_split, 't': t_train_split, 'env': envs_train_split, 'y_family': y_family_train_split}, f)
    with open(os.path.join(output_folder, 'val_data.pkl'), 'wb') as f:
        pickle.dump({'X': X_val_split, 'y': y_val_split, 't': t_val_split, 'env': envs_val_split, 'y_family': y_family_val_split}, f)
    
        print("Saved train_data.pkl and val_data.pkl")
    
    # Process test sets
    if test_list is None:
        print("\nProcessing all test sets...")
        months = _get_all_test_months()
    else:
        print(f"\nProcessing {len(test_list)} specified test months...")
        months = test_list
    
    for month in months:
        input_path = os.path.join(dataset_folder, f"{month}", "features.pkl")
        output_path = os.path.join(output_folder, f"{month}.pkl")
        if os.path.exists(input_path):
            print(f"Processing {month}...")
            process_single_file(input_path, vec, selected_feature_names, output_path)
        else:
            print(f"  Warning: Input file not found: {input_path}, skipping...")
    
    return vec, selected_feature_names

def _get_all_test_months():
    """Get list of all test months (2015-01 to 2025-06)"""
    months = []
    for year in range(2015, 2026):
        for month in range(1, 13):
            if year == 2025 and month > 6:
                break
            months.append(f"{year}-{month:02d}")
    return months

def process_all_months(vec=None, selected_feature_names=None, method='randomforest'):
    # Load vectorizer and features if not provided
    if vec is None or selected_feature_names is None:
        selected_features_path = os.path.join(save_folder, f'selected_features_{method}.txt')
        vectorizer_path = os.path.join(save_folder, f'vectorizer_{method}.pkl')
        
        if not os.path.exists(selected_features_path):
            raise FileNotFoundError(f"Selected features file not found: {selected_features_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        
        print(f"Loading selected features from {selected_features_path}...")
        with open(selected_features_path, 'r') as f:
            selected_feature_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(selected_feature_names)} selected features")
        
        print(f"Loading vectorizer from {vectorizer_path}...")
        vec = joblib.load(vectorizer_path)
        print("Vectorizer loaded successfully")
    
    # All months to process
    months = _get_all_test_months()
    
    for month in months:
        print(f"Processing {month}")
        input_path = os.path.join(dataset_folder, f"{month}", "features.pkl")
        output_path = os.path.join(save_folder, f"{month}.pkl")
        if os.path.exists(input_path):
            process_single_file(input_path, vec, selected_feature_names, output_path)
        else:
            print(f"  Warning: Input file not found: {input_path}, skipping...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset with different modes')
    parser.add_argument('--mode', type=str, default='regenerate',
                        choices=['process', 'regenerate', 'predefine'],
                        help='Processing mode')
    
    # Common arguments
    parser.add_argument('--output_folder', type=str, default='/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features_new',
                        help='Output folder for processed files')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size for processing large datasets')
    
    # Arguments for regenerate mode
    parser.add_argument('--method', type=str, default='randomforest',
                        choices=['randomforest', 'linearsvc', 'logistic', 'chi2', 'variance', 'none'],
                        help='Feature selection method (for regenerate mode)')
    parser.add_argument('--n_features', type=int, default=10000,
                        help='Number of features to select (for regenerate mode)')
    parser.add_argument('--skip_train_val', action='store_true',
                        help='Skip processing train/val sets in regenerate mode (default: process train/val)')
    
    # Arguments for process/predefine modes
    parser.add_argument('--vectorizer_file', type=str, default=None,
                        help='Path to vectorizer pkl file')
    parser.add_argument('--feature_list_file', type=str, default=None,
                        help='Path to feature list txt file (for process mode)')
    parser.add_argument('--predefined_features_file', type=str, default=None,
                        help='Path to external feature list file (for predefine mode)')
    
    # Arguments for process mode
    parser.add_argument('--input_files', type=str, nargs='+', default=None,
                        help='List of input file paths (for process mode)')
    
    # Arguments for regenerate/predefine modes
    parser.add_argument('--test_list', type=str, default=None,
                        help='a list of test months')
    parser.add_argument('--env_split_mode', type=str, default='quarter',
                        choices=['quarter', 'month', 'uniform'],
                        help='Environment split mode: "quarter" for quarterly split (Q1-Q4), "month" for monthly split (Jan-Dec), "uniform" for uniform split into n_envs environments. Default: quarter')
    parser.add_argument('--n_envs', type=int, default=4,
                        help='Number of environments for uniform split mode (only used when env_split_mode="uniform"). Default: 4')
    
    args = parser.parse_args()
    
    
    # Set default output folder
    if args.output_folder is None:
        args.output_folder = save_folder
    
    # Parse test_list if provided
    test_list = None
    
    # Call appropriate function based on mode
    if args.mode == 'process':
        if args.feature_list_file is None or args.vectorizer_file is None:
            parser.error("--feature_list_file and --vectorizer_file are required for process mode")
        if args.input_files is None:
            parser.error("--input_files is required for process mode")
        process_dataset(
            mode='process',
            feature_list_file=args.feature_list_file,
            vectorizer_file=args.vectorizer_file,
            input_files=args.input_files,
            output_folder=args.output_folder
        )
    elif args.mode == 'regenerate':
        process_train_val = not args.skip_train_val  # Default is True unless --skip_train_val is set
        process_dataset(
            mode='regenerate',
            method=args.method,
            n_features=args.n_features,
            batch_size=args.batch_size,
            vectorizer_file=args.vectorizer_file,
            output_folder=args.output_folder,
            process_train_val=process_train_val,
            test_list=test_list,
            env_split_mode=args.env_split_mode,
            n_envs=args.n_envs
        )
    elif args.mode == 'predefine':
        if args.predefined_features_file is None or args.vectorizer_file is None:
            parser.error("--predefined_features_file and --vectorizer_file are required for predefine mode")
        process_dataset(
            mode='predefine',
            predefined_features_file=args.predefined_features_file,
            vectorizer_file=args.vectorizer_file,
            output_folder=args.output_folder,
            batch_size=args.batch_size,
            test_list=test_list,
            env_split_mode=args.env_split_mode,
            n_envs=args.n_envs
        )
