import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

def load_single_month_data(file_path):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"keys: {data.keys()}")
    x_data = data['X']
    y_data = data['y']

    print(f"X shape: {x_data.shape}, y shape: {y_data.shape}, y distribution: {Counter(y_data)}")

    return x_data, y_data

def load_train_data(file_path, chunk_size=10000):
    """
    Load and split data in chunks to manage memory efficiently
    
    Args:
        file_path: path to the pickle file
        chunk_size: size of each chunk to load
    
    Returns:
        x_train, x_val, y_train, y_val: split training and validation data
    """
    from scipy import sparse
    
    # Initialize empty lists for collecting chunks
    x_train_list, x_val_list = [], []
    y_train_list, y_val_list = [], []
    
    # First pass: get total size
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        total_size = len(data['y'])
        # Check if X is sparse
        is_sparse = sparse.issparse(data['X'])

    print(f"Total size: {total_size}")
    print(f"Data is sparse: {is_sparse}")
    
    # Second pass: process chunks
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
        for start_idx in tqdm(range(0, total_size, chunk_size)):
            end_idx = min(start_idx + chunk_size, total_size)
            
            # Extract chunk
            chunk_x = data['X'][start_idx:end_idx]
            chunk_y = data['y'][start_idx:end_idx]

            print(f"Loaded chunk {chunk_x.shape}, {chunk_y.shape}") 
            
            # Split chunk into train and validation
            x_train_chunk, x_val_chunk, y_train_chunk, y_val_chunk = train_test_split(
                chunk_x, chunk_y,
                test_size=0.2,
                random_state=42,
                stratify=chunk_y
            )
            
            # Append chunks to lists
            x_train_list.append(x_train_chunk)
            x_val_list.append(x_val_chunk)
            y_train_list.append(y_train_chunk)
            y_val_list.append(y_val_chunk)
            
            print(f"Processed chunk {start_idx//chunk_size + 1}/{(total_size-1)//chunk_size + 1}")
    
    # Concatenate all chunks properly based on data type
    if is_sparse:
        # Use sparse vstack for sparse matrices
        x_train = sparse.vstack(x_train_list)
        x_val = sparse.vstack(x_val_list)
    else:
        # Use numpy concatenate for dense arrays
        x_train = np.concatenate(x_train_list, axis=0)
        x_val = np.concatenate(x_val_list, axis=0)
    
    # Labels are always dense arrays
    y_train = np.concatenate(y_train_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    
    print(f"Final shapes - Train: {x_train.shape}, Validation: {x_val.shape}")
    return x_train, x_val, y_train, y_val


def load_train_data_with_env(file_path, chunk_size=10000):
    """
    Load and split data in chunks to manage memory efficiently
    
    Args:
        file_name: name of the pickle file
        data_folder: path to the data folder
        chunk_size: size of each chunk to load
    
    Returns:
        x_train, x_val, y_train, y_val: split training and validation data
    """
    # Initialize empty lists for collecting chunks
    x_train_list, x_val_list = [], []
    y_train_list, y_val_list = [], []
    env_train_list, env_val_list = [], []
    
    # file_path = os.path.join(data_folder, file_name)
    
    
    # First pass: get total size
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        total_size = len(data['y'])
    
    # Second pass: process chunks
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
        for start_idx in tqdm(range(0, total_size, chunk_size)):
            end_idx = min(start_idx + chunk_size, total_size)
            
            # Extract chunk
            chunk_x = data['X'][start_idx:end_idx]
            chunk_y = data['y'][start_idx:end_idx]
            chunk_env = data['env'][start_idx:end_idx]
            
            # Split chunk into train and validation
            x_train_chunk, x_val_chunk, y_train_chunk, y_val_chunk,env_train_chunk, env_val_chunk = train_test_split(
                chunk_x, chunk_y,chunk_env,
                test_size=0.2,
                random_state=42,
                stratify=chunk_y
            )
            
            # Append chunks to lists
            x_train_list.append(x_train_chunk)
            x_val_list.append(x_val_chunk)
            y_train_list.append(y_train_chunk)
            y_val_list.append(y_val_chunk)
            env_train_list.append(env_train_chunk)
            env_val_list.append(env_val_chunk)
            
            print(f"Processed chunk {start_idx//chunk_size + 1}/{(total_size-1)//chunk_size + 1}")
    
    # Concatenate all chunks
    x_train = np.concatenate(x_train_list, axis=0)
    x_val = np.concatenate(x_val_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    env_train = np.concatenate(env_train_list, axis=0)
    env_val = np.concatenate(env_val_list, axis=0)
    
    print(f"Final shapes - Train: {x_train.shape}, Validation: {x_val.shape}")
    return x_train, x_val, y_train, y_val, env_train, env_val


def generate_month_list():
    test_year_list = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    month_list = [f"{year}-{month}" for year in test_year_list for month in months]

    return month_list

def load_and_process_single_month(file_path, year):
    x_data, y_data = load_single_month_data(file_path)
    y_data = np.where(y_data == 1, 1, 0)

    env_value = {'2017': 0, '2018': 1, '2019': 2}[year]
    env_data = np.full_like(y_data, env_value)

    print(f"Loaded {file_path} -> X shape: {x_data.shape}, y shape: {y_data.shape}, y distribution: {Counter(y_data)}")

    return x_data, y_data, env_data

def allocate_env_label_and_save(data_folder):
    train_year_list = ['2017', '2018', '2019']
    month_list = [f"{i:02d}" for i in range(1, 13)]  
    
    file_paths = [
        (os.path.join(data_folder, f"{year}-{month}.pkl"), year)
        for year in train_year_list for month in month_list
    ]

    all_x_data, all_y_data, all_env_data = [], [], []

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda args: load_and_process_single_month(*args), file_paths))

    for x_data, y_data, env_data in results:
        all_x_data.append(x_data)
        all_y_data.append(y_data)
        all_env_data.append(env_data)


    all_x_data = np.vstack(all_x_data)  
    all_y_data = np.hstack(all_y_data)  
    all_env_data = np.hstack(all_env_data)

    print(f"Final X shape: {all_x_data.shape}, y shape: {all_y_data.shape}, env shape: {all_env_data.shape}")

    save_path = os.path.join(data_folder, "all_train_features_with_env.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            'X': all_x_data,
            'y': all_y_data,
            'env': all_env_data
        }, f)

    print(f"Data saved to {save_path}")


def get_train_dataset_envs(train_data_path, type='year'):
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    x_data = data['X']
    y_data = data['y']
    t_data = data['t']
    
    # Convert "YYYY-MM" strings to NumPy array of datetime objects
    year_month_train = np.array([datetime.strptime(t, "%Y-%m") for t in t_data])

    if type == 'year':
        # Map years directly to labels using NumPy's vectorized operations
        env_mapping = {2017: 0, 2018: 1, 2019: 2}
        years = np.array([date.year for date in year_month_train])
        envs = np.vectorize(env_mapping.get)(years, -1)  # Assign -1 if year not found
        print(Counter(envs))

    elif type == 'month':
        # Use NumPy vectorized operation for month extraction
        envs = np.array([date.month - 1 for date in year_month_train])
        print(Counter(envs))

    elif type == 'quarter':
        # Compute quarter directly using NumPy vectorized operation
        envs = (np.array([date.month for date in year_month_train]) - 1) // 3
        print(Counter(envs))

    else:
        raise ValueError("Invalid type. Choose from 'year', 'month', or 'quarter'.")


    x_train, x_val, y_train, y_val, env_train, env_val = train_test_split(x_data, y_data, envs, test_size=0.2, random_state=42, stratify=y_data)

    return x_train, x_val, y_train, y_val, env_train, env_val