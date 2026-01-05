import sys
sys.path.append('/root/malware/ELSA')
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from sklearn.utils import shuffle
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix
  
import pickle 
from sklearn.metrics import hinge_loss


data_folder = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features_old"
save_folder = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt"
result_folder = "/cs/academic/phd3/xinrzhen/xinran/SaTML/results"


def load_feature_names():
    with open(os.path.join(save_folder, f'selected_features_randomforest.txt'), 'r') as f:
        feature_names = f.read().splitlines()

    return feature_names

def creat_t_stability():
    train_path = os.path.join(data_folder, "train_data.pkl")
    with open(train_path, 'rb') as f:
        data = pickle.load(f)
    print(f"keys: {data.keys()}")
    x_data = data['X']
    y_data = data['y']
    t_data = data['t']

    x_train, x_val, y_train, y_val, t_train, t_val = train_test_split(x_data, y_data, t_data, test_size=0.2, random_state=0, stratify=y_data)

    date_array_dt = pd.to_datetime(t_train, errors='coerce')
    date_series = pd.Series(date_array_dt, index=range(len(date_array_dt)))
    groups = date_series.groupby([date_series.dt.year, date_series.dt.month])
    
    # Get sorted time periods to ensure temporal order
    time_periods = sorted(groups.groups.keys())
    T = len(time_periods)
    
    if T == 0:
        raise ValueError("No valid time periods found in the data")

    d = x_train.shape[1]

    svm = LinearSVC(max_iter=10000)
    svm.fit(x_train, y_train)
    weights = svm.coef_[0]
    bias = svm.intercept_[0]

    y_val_pred = svm.predict(x_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # feature_names = load_feature_names(data_folder)

    M = np.zeros((d, T))

    col = 0
    # Iterate through sorted time periods to maintain temporal order
    for (year, month) in time_periods:
        group = groups.get_group((year, month))
        index = group.index.tolist()
        X = x_train[index]
        y_label = y_train[index]
        mask = y_label == 1
        X_malware = X[mask]
        print(f"Year: {year}, Month: {month}, index shape: {len(index)}, X shape: {X.shape}, X_malware shape: {X_malware.shape}, y_label shape: {y_label.shape}")
        
        # Fix: Always fill M matrix, even if no malware samples
        # This ensures M matrix columns correspond to actual time periods
        if X_malware.shape[0] == 0:
            # Use zeros when no malware samples (alternative: could use mean of all samples in that period)
            M[:, col] = 0.0
            print(f"  Warning: No malware samples for {year}-{month:02d}, using zeros")
        else:
            M[:, col] = np.mean(X_malware, axis=0)
        
        col += 1

    print(f"M shape: {M.shape}, T={T}")
    
    slopes = np.zeros(d)
    for j in range(d):
        reg = LinearRegression()
        # Fix: Use sequential time indices (0, 1, 2, ..., T-1) since we've ensured
        # M matrix columns are in temporal order and all columns are filled
        reg.fit(np.arange(T).reshape(-1, 1), M[j, :])
        slopes[j] = reg.coef_[0]

    T_stability = weights * slopes

    t_stability_df = pd.DataFrame({
    'Index': [i for i in range(d)],
    # 'Feature': [feature_names[i] for i in range(d)],
    'Weight (w_j)': weights,
    'Slope (m_j)': slopes,
    'T-stability': T_stability
    })

    t_stability_df = t_stability_df.sort_values(by='T-stability', ascending=True)
    print(t_stability_df.head(10))
    print(f"max T-stability: {t_stability_df['T-stability'].max()}, min T-stability: {t_stability_df['T-stability'].min()}")

    with open(os.path.join(save_folder,f"t_stability.pkl"), "wb") as f:
        pickle.dump(t_stability_df, f)
    print(f"t_stability saved to {os.path.join(save_folder,f't_stability.pkl')}")


# Function to modulate learning rate with cosine annealing
def cosine_annealing(t, T=1000):
    return 0.5 * (1 + np.cos(np.pi * t / T))

# Custom training loop for SVM with constrained weights (SVM-CB)
def svm_cb_train(X, y, t_stability, n_f, r, num_iterations=1000, 
                           initial_lr=7e-5, lambda_reg=0.01, s_func=None, 
                           batch_size=256, early_stop_patience=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    eval_interval = 50
    
    # Pre-compute unstable features indices
    unstable_features_idx = np.argsort(t_stability)[:n_f]
    
    # For early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    # Pre-shuffle data
    X, y = shuffle(X, y, random_state=42)
    
    # Track time
    start_time = time.time()
    
    for t in tqdm(range(num_iterations)):
        if s_func:
            learning_rate_t = initial_lr * s_func(t)
        else:
            learning_rate_t = initial_lr
        
        # Process in mini-batches
        indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        
        # Vectorized margin calculation
        margins = y_batch * (np.dot(X_batch, w) + b)
        
        # Identify samples with margin < 1
        violated_idx = margins < 1
        
        # Vectorized gradient calculation
        grad_w = lambda_reg * w  # Start with regularization
        if np.any(violated_idx):
            X_violated = X_batch[violated_idx]
            y_violated = y_batch[violated_idx]
            grad_w -= np.sum(y_violated[:, np.newaxis] * X_violated, axis=0)
            grad_b = -np.sum(y_violated)
        else:
            grad_b = 0
        
        # Loss calculation
        hinge_loss_val = np.sum(np.maximum(0, 1 - margins))
        reg_loss = (lambda_reg / 2) * np.dot(w, w)
        total_loss = (hinge_loss_val / batch_size) + reg_loss
        
        # Update weights and bias
        w -= learning_rate_t * grad_w / batch_size
        b -= learning_rate_t * grad_b / batch_size
        
        # Apply constraints on unstable features
        w[unstable_features_idx] = np.clip(w[unstable_features_idx], -r, r)
        
        # Reporting and early stopping
        if t % eval_interval == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {t}: Loss={total_loss:.6f}, Time elapsed: {elapsed:.2f}s")
            
            # Early stopping check
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
                best_w, best_b = w.copy(), b
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at iteration {t}")
                    return best_w, best_b
    
    return w, b

def svm_cb_predict(X, w, b):
    """
    Perform inference using the trained SVM-CB model.
    
    Parameters:
    - X: Test data (n_samples, n_features)
    - w: Trained weights
    - b: Trained bias
    
    Returns:
    - predictions: Predicted labels (0 or 1)
    """
    predictions = np.sign(np.dot(X, w) + b)
    pred_label = np.where(predictions == -1, 0, 1)

    return pred_label


def retrain_svm(ts_path):
    # load t stability
    
    with open(ts_path, "rb") as f:
        t_stability_df = pickle.load(f)
    train_path = os.path.join(data_folder, "train_data.pkl")
    with open(train_path, 'rb') as f:
        data = pickle.load(f)
    print(f"keys: {data.keys()}")
    x_data = data['X']
    y_data = data['y']
    t_data = data['t']

    # transfer to dense array
    x_data = x_data.toarray()

    x_train, x_val, y_train, y_val, t_train, t_val = train_test_split(x_data, y_data, t_data, test_size=0.2, random_state=0, stratify=y_data)
    y_train_transform = np.where(y_train == 0, -1, 1)

    n_f = 200  # Number of unstable features to constrain
    r = 0.2  # Bound for the weights of unstable features
    num_iterations = 5000
    eta_0 = 0.01


    svm = LinearSVC(max_iter=10000)
    svm.fit(x_train, y_train)
    decision_function = svm.decision_function(x_train)
    # Calculate the hinge loss using the decision function and true labels
    loss = hinge_loss(y_train, decision_function)
    print("Hinge loss for training set:", loss)

    # unstable_feature_indices = np.argsort(t_stability_df['T-stability'].values)[:n_f]
    # W_f = unstable_feature_indices.astype(int)
    t_stability = t_stability_df['T-stability'].values

    s_func = lambda t: cosine_annealing(t, T=num_iterations)
    w, b =  svm_cb_train(x_train, y_train_transform, t_stability , n_f, r, num_iterations=num_iterations, initial_lr=eta_0, s_func=s_func)
    model_save_path = os.path.join(save_folder, f"svm_model_{num_iterations}.pkl")
    with open(model_save_path, 'wb') as f:
        pickle.dump({'w': w, 'b': b}, f)
    print(f"Model saved to {model_save_path}")

    with open(model_save_path, 'rb') as f:
        model = pickle.load(f)
        w = model['w']
        b = model['b']
    print(f"Model loaded from {model_save_path}")

    y_val_pred = svm_cb_predict(x_val, w, b)
    f1 = f1_score(y_val, y_val_pred, average='macro')

    return w, b, f1

