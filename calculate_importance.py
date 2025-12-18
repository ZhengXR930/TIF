import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from captum.attr import IntegratedGradients
import pickle
import json
from scipy import sparse
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from collections import Counter
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


family_dict_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/combine_drebin/family_dict.json"
feature_names_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features/selected_features.txt"
processed_feature_folder = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features"
train_data_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features/train_data.pkl"
figures_folder = "/cs/academic/phd3/xinrzhen/xinran/SaTML/figures"

class SimpleDataset(Dataset):
    """Simple dataset wrapper for numpy arrays"""
    def __init__(self, X, y):
        if sparse.issparse(X):
            self.X = torch.FloatTensor(X.toarray())
        else:
            self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_feature_names(feature_file_path):
    """Load feature names from text file"""
    with open(feature_file_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    return feature_names


def binomial_noise(data, flip_prob=0.1):
    device = data.device
    noise = torch.bernoulli(torch.full_like(data, fill_value=flip_prob)).to(device)
    noisy_data = torch.abs(data - noise)  
    return noisy_data


def random_baseline_integrated_gradients(input_dataset, model, device, target_index, steps=50, num_random_trials=5, train_flag=True):

    model.to(device)
    model.eval()
    
    # Create a wrapper function that returns only the prediction outputs (not the tuple)
    def model_wrapper(input_tensor):
        outputs, _ = model(input_tensor)
        return outputs
    
    integrated_gradients = IntegratedGradients(model_wrapper)

    accumulated_grads = 0

    batch_size = 64

    input_loader = DataLoader(dataset=input_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for index, (test_batch, test_label, *_) in enumerate(input_loader):
        test_batch = test_batch.to(torch.float32).to(device)
        batch_grads = 0

        for num in range(num_random_trials):
            baseline_batch = torch.zeros_like(test_batch).to(device)
            noisy_test_batch = binomial_noise(test_batch)
            int_grads_batch,_ = integrated_gradients.attribute(noisy_test_batch, baseline_batch,target = target_index,return_convergence_delta=True)

            batch_grads += int_grads_batch

        batch_grads /= num_random_trials
        accumulated_grads += batch_grads.sum(dim=0)  

    # Calculate average per sample (not per dataset size)
    num_samples = len(input_loader.dataset)
    accumulated_grads = accumulated_grads / num_samples
    print(f"type of accumulated_grads: {type(accumulated_grads)}")

    accumulated_grads = accumulated_grads.detach().cpu().numpy()
    min_val = np.min(accumulated_grads)
    max_val = np.max(accumulated_grads)
    
    # Avoid division by zero
    if max_val - min_val > 1e-10:
        normalized_importances = (accumulated_grads - min_val) / (max_val - min_val)
    else:
        normalized_importances = accumulated_grads

    return normalized_importances


def process_result(feature_importances, feature_names, top_k=50):
    """Process importance scores and return top K feature names"""
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_importances = feature_importances[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    
    # Return top K
    return sorted_feature_names[:top_k], sorted_importances[:top_k]


def select_existfamily_by_file(file_path, select_family):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    y_family = data['y_family']
    env = data['env']

    with open(family_dict_path, 'r') as f:
        family_dict = json.load(f)
    
    select_family_encoded = [family_dict[family] for family in select_family]

    # First select by family (encoded ids)
    selected_idx = np.where(np.isin(y_family, select_family_encoded))
    X_selected = X[selected_idx]
    y_selected = y[selected_idx]
    y_family_selected = y_family[selected_idx]
    env_selected = env[selected_idx]

    # All families are malware; if any benign (label 0) appear, remove them
    if len(y_selected) > 0:
        malware_mask = (y_selected == 1)
        if malware_mask.sum() != len(y_selected):
            X_selected = X_selected[malware_mask]
            y_selected = y_selected[malware_mask]
            y_family_selected = y_family_selected[malware_mask]
            env_selected = env_selected[malware_mask]

    return X_selected, y_selected, env_selected, y_family_selected

def select_existfamily_for_test(select_family):
    test_list = ['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
                '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12',
                '2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
                '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12',
                '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
                '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12',
                '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
                '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
                '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11']
    # Handle both string family names and numeric family IDs
    # If select_family contains strings, map via family_dict.
    # If it already contains numeric IDs (like diff_test_family), use them directly.
    with open(family_dict_path, 'r') as f:
        family_dict = json.load(f)

    if len(select_family) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    first_elem = select_family[0]
    if isinstance(first_elem, (int, np.integer)):
        # Already encoded family IDs
        select_family_encoded = list(select_family)
    else:
        # Family names → encoded IDs
        select_family_encoded = []
        for family in select_family:
            if family in family_dict:
                select_family_encoded.append(family_dict[family])
            else:
                print(f"Warning: family '{family}' not found in family_dict, skipping.")

    if len(select_family_encoded) == 0:
        print("Warning: no valid families found after encoding.")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Lists to collect data from all months
    X_list, y_list, env_list, y_family_list = [], [], [], []
    
    for month in test_list:
        file_path = os.path.join(processed_feature_folder, f"{month}.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        X = data['X']
        y = data['y']
        y_family = data['y_family']
        env = data['env']

        # Select samples belonging to the target family in this month
        selected_idx = np.where(np.isin(y_family, select_family_encoded))
        if selected_idx[0].size == 0:
            continue

        X_sel = X[selected_idx]
        y_sel = y[selected_idx]
        env_sel = env[selected_idx]
        y_family_sel = y_family[selected_idx]

        X_list.append(X_sel)
        y_list.append(y_sel)
        env_list.append(env_sel)
        y_family_list.append(y_family_sel)
    # If no samples found, return empty arrays
    if len(X_list) == 0:
        return np.array([]), np.array([]), np.array([])

    # Concatenate across months, handling sparse and dense matrices
    if sparse.issparse(X_list[0]):
        X_selected = sparse.vstack(X_list)
    else:
        X_selected = np.concatenate(X_list, axis=0)

    y_selected = np.concatenate(y_list, axis=0)
    env_selected = np.concatenate(env_list, axis=0)
    y_family_selected = np.concatenate(y_family_list, axis=0)

    # All families are malware; if any benign (label 0) appear, remove them
    if len(y_selected) > 0:
        malware_mask = (y_selected == 1)
        if malware_mask.sum() != len(y_selected):
            X_selected = X_selected[malware_mask]
            y_selected = y_selected[malware_mask]
            env_selected = env_selected[malware_mask]
            y_family_selected = y_family_selected[malware_mask]
    return X_selected, y_selected, env_selected, y_family_selected

def get_all_test_data():
    test_list = ['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
                '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12',
                '2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
                '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12',
                '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
                '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12',
                '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
                '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
                '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11']
    X_list, y_list, env_list, y_family_list = [], [], [], []
    for month in test_list:
        file_path = os.path.join(processed_feature_folder, f"{month}.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        X = data['X']
        y = data['y']
        y_family = data['y_family']
        env = data['env']

        X_list.append(X)
        y_list.append(y)
        env_list.append(env)
        y_family_list.append(y_family)

    # Handle empty case
    if len(X_list) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Concatenate across months, handling sparse and dense matrices
    if sparse.issparse(X_list[0]):
        # X is sparse for each month
        X_selected = sparse.vstack(X_list)
    else:
        # X is dense numpy array
        X_selected = np.concatenate(X_list, axis=0)

    y_selected = np.concatenate(y_list, axis=0)
    env_selected = np.concatenate(env_list, axis=0)
    y_family_selected = np.concatenate(y_family_list, axis=0)

    return X_selected, y_selected, env_selected, y_family_selected


def get_representations_train_vs_unseen(
    model_type,
    model_path,
    X_train,
    y_train,
    y_family_train,
    X_test,
    y_test,
    y_family_test,
    unseen_family_ids,
    train_exclude_family=37,
    sample_size=2000,
):
    """
    Get encoder representations for:
      - Training families (excluding a given benign family, default id=37)
      - Unseen families in the test set (family ids in unseen_family_ids)

    Returns:
        reps_train_sampled, reps_unseen_sampled
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Filter training set: remove benign family 37 and ensure malware label y==1
    train_mask = (y_family_train != train_exclude_family) & (y_train == 1)
    X_train_filt = X_train[train_mask]
    y_train_filt = y_train[train_mask]
    y_family_train_filt = y_family_train[train_mask]

    # 2) Filter test set: unseen families and malware label y==1
    unseen_mask = np.isin(y_family_test, unseen_family_ids) & (y_test == 1)
    X_unseen = X_test[unseen_mask]
    y_unseen = y_test[unseen_mask]
    y_family_unseen = y_family_test[unseen_mask]

    # Guard against no data
    if (hasattr(X_train_filt, "shape") and X_train_filt.shape[0] == 0) or (
        not hasattr(X_train_filt, "shape") and len(X_train_filt) == 0
    ):
        print("Warning: no training samples after filtering; cannot compute representations.")
        return np.empty((0, 0)), np.empty((0, 0))
    if (hasattr(X_unseen, "shape") and X_unseen.shape[0] == 0) or (
        not hasattr(X_unseen, "shape") and len(X_unseen) == 0
    ):
        print("Warning: no unseen test samples after filtering; cannot compute representations.")
        return np.empty((0, 0)), np.empty((0, 0))

    # 3) Randomly sample up to sample_size from each
    rng = np.random.default_rng(42)

    def _sample_indices(n, k):
        k = min(k, n)
        return rng.choice(n, size=k, replace=False)

    n_train = X_train_filt.shape[0] if hasattr(X_train_filt, "shape") else len(X_train_filt)
    n_unseen = X_unseen.shape[0] if hasattr(X_unseen, "shape") else len(X_unseen)

    idx_train = _sample_indices(n_train, sample_size)
    idx_unseen = _sample_indices(n_unseen, sample_size)

    if sparse.issparse(X_train_filt):
        X_train_sampled = X_train_filt[idx_train]
    else:
        X_train_sampled = X_train_filt[idx_train]

    if sparse.issparse(X_unseen):
        X_unseen_sampled = X_unseen[idx_unseen]
    else:
        X_unseen_sampled = X_unseen[idx_unseen]

    # 4) Load model and compute representations
    input_size = X_train.shape[1]
    model = _load_model_for_representation(
        model_type=model_type,
        model_path=model_path,
        input_size=input_size,
        device=device,
    )

    reps_train_sampled = _compute_representations(X_train_sampled, model, device)
    reps_unseen_sampled = _compute_representations(X_unseen_sampled, model, device)

    print(
        f"[{model_type}] train reps shape: {reps_train_sampled.shape}, "
        f"unseen reps shape: {reps_unseen_sampled.shape}"
    )

    return reps_train_sampled, reps_unseen_sampled


def plot_2d_embeddings_two_groups(
    reps_train,
    reps_unseen,
    model_name="TIF",
    method="umap",
    save_path=None,
):
    """
    Plot 2D embeddings (UMAP or t-SNE) for two distributions:
      1) training families (excluding benign 37)
      2) unseen test families
    """
    import matplotlib.pyplot as plt

    all_reps = []
    group_ids = []

    if reps_train is not None and reps_train.shape[0] > 0:
        all_reps.append(reps_train)
        group_ids.extend([0] * reps_train.shape[0])

    if reps_unseen is not None and reps_unseen.shape[0] > 0:
        all_reps.append(reps_unseen)
        group_ids.extend([1] * reps_unseen.shape[0])

    if len(all_reps) == 0:
        print("No representations to plot (two-group).")
        return

    all_reps = np.concatenate(all_reps, axis=0)
    group_ids = np.array(group_ids)

    # Dimensionality reduction
    if method.lower() == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, init="random", random_state=None)
        coords = reducer.fit_transform(all_reps)
        title = f"{model_name} - UMAP (Train vs Unseen Families)"
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = tsne.fit_transform(all_reps)
        title = f"{model_name} - t-SNE (Train vs Unseen Families)"

    plt.figure(figsize=(8, 6))
    colors = {0: "tab:blue", 1: "tab:red"}
    labels_map = {
        0: "Train families (except 37)",
        1: "Unseen test families",
    }

    for gid in np.unique(group_ids):
        mask = group_ids == gid
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.7,
            c=colors.get(gid, "gray"),
            label=labels_map.get(gid, f"group_{gid}"),
        )

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.tight_layout()

    # Always save to figures folder by default
    if save_path is None:
        os.makedirs(figures_folder, exist_ok=True)
        safe_model_name = model_name.replace(" ", "_").replace("/", "_")
        safe_method = method.lower()
        save_path = os.path.join(
            figures_folder, f"{safe_model_name}_{safe_method}_train_unseen.png"
        )

    plt.savefig(save_path, dpi=300)
    print(f"Saved two-group plot to {save_path}")


def _load_model_for_representation(model_type, model_path, input_size, device):
    """
    Helper to load a model for getting representations.
    model_type: 'tif_stage1', 'tif_stage2', or 'deepdrebin'
    """
    if model_type in ('tif', 'tif_stage1'):
        from stage1_trainer import St1ModelTrainer
        from model import DrebinMLP_IRM
        model = St1ModelTrainer.load_model(
            model_path=model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size,
            device=device
        )
    elif model_type == 'tif_stage2':
        from stage2_trainer import St2ModelTrainer
        from model import DrebinMLP_IRM
        model, _ = St2ModelTrainer.load_model(
            model_path=model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size
        )
        model = model.to(device)
    elif model_type == 'deepdrebin':
        from trainer import ModelTrainer
        from model import DrebinMLP
        model = ModelTrainer.load_model(
            model_path=model_path,
            model_class=DrebinMLP,
            input_size=input_size,
            device=device
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval()
    return model


def _compute_representations(X, model, device, batch_size=512):
    """
    Compute encoder representations for all samples in X.
    Returns: numpy array of shape (N, emb_dim)
    """
    from model import DrebinMLP, DrebinMLP_IRM  # for isinstance checks if needed

    dataset = SimpleDataset(X, np.zeros(X.shape[0]))  # labels not used
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            # Both DrebinMLP and DrebinMLP_IRM return (prob, feature)
            _, features = model(inputs)
            all_features.append(features.detach().cpu().numpy())

    if len(all_features) == 0:
        return np.empty((0, getattr(model, "emb_dim", 0)))

    return np.concatenate(all_features, axis=0)


def get_family_representations_for_model(
    model_type,
    model_path,
    train_family_name="airpush",
    test_family_names=("airpush", "hiddad"),
):
    """
    Get encoder representations for:
      1) train Airpush family
      2) test Airpush family
      3) test Hiddad family
    for a given model.

    Returns:
        reps_train_airpush, reps_test_airpush, reps_test_hiddad
        (each is a numpy array of shape (N_i, emb_dim))
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load train airpush
    X_train_airpush, _, _, _ = select_existfamily_by_file(
        train_data_path, [train_family_name]
    )

    # 2) Load test airpush & hiddad across months
    X_test_airpush, _, _, _ = select_existfamily_for_test([test_family_names[0]])
    X_test_hiddad, _, _, _ = select_existfamily_for_test([test_family_names[1]])

    # Determine input size from train data
    if sparse.issparse(X_train_airpush):
        input_size = X_train_airpush.shape[1]
    else:
        input_size = X_train_airpush.shape[1]

    model = _load_model_for_representation(
        model_type=model_type,
        model_path=model_path,
        input_size=input_size,
        device=device,
    )

    reps_train_airpush = _compute_representations(X_train_airpush, model, device)
    reps_test_airpush = _compute_representations(X_test_airpush, model, device)
    reps_test_hiddad = _compute_representations(X_test_hiddad, model, device)

    return reps_train_airpush, reps_test_airpush, reps_test_hiddad


def plot_2d_embeddings(
    reps_train_airpush,
    reps_test_airpush,
    reps_test_hiddad,
    model_name="TIF",
    method="tsne",
    save_path=None,
):
    """
    Plot 2D embeddings (UMAP or t-SNE) for three distributions:
      1) train airpush
      2) test airpush
      3) test hiddad
    """
    import matplotlib.pyplot as plt

    # Combine all for joint projection
    all_reps = []
    labels = []
    group_ids = []

    if reps_train_airpush is not None and reps_train_airpush.shape[0] > 0:
        all_reps.append(reps_train_airpush)
        labels.extend(["train_airpush"] * reps_train_airpush.shape[0])
        group_ids.extend([0] * reps_train_airpush.shape[0])

    if reps_test_airpush is not None and reps_test_airpush.shape[0] > 0:
        all_reps.append(reps_test_airpush)
        labels.extend(["test_airpush"] * reps_test_airpush.shape[0])
        group_ids.extend([1] * reps_test_airpush.shape[0])

    if reps_test_hiddad is not None and reps_test_hiddad.shape[0] > 0:
        all_reps.append(reps_test_hiddad)
        labels.extend(["test_hiddad"] * reps_test_hiddad.shape[0])
        group_ids.extend([2] * reps_test_hiddad.shape[0])

    if len(all_reps) == 0:
        print("No representations to plot.")
        return

    all_reps = np.concatenate(all_reps, axis=0)
    group_ids = np.array(group_ids)

    # Dimensionality reduction
    if method.lower() == "umap" and HAS_UMAP:
        # Use random init and no fixed random_state to avoid spectral and n_jobs warnings
        reducer = umap.UMAP(n_components=2, init="random", random_state=None)
        coords = reducer.fit_transform(all_reps)
        title = f"{model_name} - UMAP of representations"
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = tsne.fit_transform(all_reps)
        title = f"{model_name} - t-SNE of representations"

    plt.figure(figsize=(8, 6))
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    labels_map = {0: "Train - Airpush", 1: "Test - Airpush", 2: "Test - Hiddad"}

    for gid in np.unique(group_ids):
        mask = group_ids == gid
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.7,
            c=colors.get(gid, "gray"),
            label=labels_map.get(gid, f"group_{gid}"),
        )

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.tight_layout()

    # Always save to figures folder by default
    if save_path is None:
        # Ensure figures directory exists
        os.makedirs(figures_folder, exist_ok=True)
        # Build a safe filename from model name and method
        safe_model_name = model_name.replace(" ", "_").replace("/", "_")
        safe_method = method.lower()
        save_path = os.path.join(
            figures_folder, f"{safe_model_name}_{safe_method}_repr.png"
        )

    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

def calculate_importance(X_selected, y_selected, env_selected, model_path, feature_names_path=None, top_k=50):
    """
    Calculate feature importance for selected data
    
    Args:
        X_selected: Feature matrix (numpy array or sparse matrix)
        y_selected: Labels (numpy array)
        env_selected: Environment labels (numpy array)
        model_path: Path to the trained model
        feature_names_path: Path to feature names file (defaults to module-level path)
        top_k: Number of top features to return
    
    Returns:
        top_feature_names: List of top K feature names
        top_importances: List of top K importance scores
    """
    # Use default feature names path if not provided
    if feature_names_path is None:
        feature_names_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features/selected_features.txt"
    
    # Guard against empty selection
    if X_selected is None:
        print("Warning: X_selected is None.")
        return [], []
    num_samples = X_selected.shape[0] if hasattr(X_selected, "shape") else len(X_selected)
    if num_samples == 0:
        print("Warning: no samples provided to calculate_importance; skipping IG.")
        return [], []

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model using St1ModelTrainer.load_model to match main.py pattern
    from stage1_trainer import St1ModelTrainer
    from model import DrebinMLP_IRM
    
    input_size = X_selected.shape[1]
    model = St1ModelTrainer.load_model(
        model_path=model_path,
        model_class=DrebinMLP_IRM,
        input_size=input_size,
        device=device
    )
    
    # Create dataset from numpy arrays
    dataset = SimpleDataset(X_selected, y_selected)
    
    # Calculate importance
    importance = random_baseline_integrated_gradients(dataset, model, device, target_index=1)
    
    # Load feature names
    feature_names = load_feature_names(feature_names_path)
    
    # Verify feature count matches
    if len(feature_names) != len(importance):
        print(f"Warning: Feature count mismatch! Feature names: {len(feature_names)}, Importance scores: {len(importance)}")
        print(f"Using min length: {min(len(feature_names), len(importance))}")
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]
    
    # Get top K features
    top_feature_names, top_importances = process_result(importance, feature_names, top_k=top_k)
    
    return top_feature_names, top_importances


def calculate_importance_deepdrebin(X_selected, y_selected, env_selected, model_path, feature_names_path=None, top_k=50):
    """
    Calculate feature importance for selected data using DeepDrebin model
    
    Args:
        X_selected: Feature matrix (numpy array or sparse matrix)
        y_selected: Labels (numpy array)
        env_selected: Environment labels (numpy array)
        model_path: Path to the trained DeepDrebin model
        feature_names_path: Path to feature names file (defaults to module-level path)
        top_k: Number of top features to return
    
    Returns:
        top_feature_names: List of top K feature names
        top_importances: List of top K importance scores
    """
    # Guard against empty selection
    if X_selected is None:
        print("Warning: X_selected is None.")
        return [], []
    num_samples = X_selected.shape[0] if hasattr(X_selected, "shape") else len(X_selected)
    if num_samples == 0:
        print("Warning: no samples provided to calculate_importance_deepdrebin; skipping IG.")
        return [], []

    # Use default feature names path if not provided
    if feature_names_path is None:
        feature_names_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features/selected_features.txt"
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model using ModelTrainer.load_model for DeepDrebin
    from trainer import ModelTrainer
    from model import DrebinMLP
    
    input_size = X_selected.shape[1]
    model = ModelTrainer.load_model(
        model_path=model_path,
        model_class=DrebinMLP,
        input_size=input_size,
        device=device
    )
    
    # Create dataset from numpy arrays
    dataset = SimpleDataset(X_selected, y_selected)
    
    # Calculate importance
    importance = random_baseline_integrated_gradients(dataset, model, device, target_index=1)
    
    # Load feature names
    feature_names = load_feature_names(feature_names_path)
    
    # Verify feature count matches
    if len(feature_names) != len(importance):
        print(f"Warning: Feature count mismatch! Feature names: {len(feature_names)}, Importance scores: {len(importance)}")
        print(f"Using min length: {min(len(feature_names), len(importance))}")
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]
    
    # Get top K features
    top_feature_names, top_importances = process_result(importance, feature_names, top_k=top_k)
    
    return top_feature_names, top_importances


def compare_importance(tif_feature_names, tif_importances, deepdrebin_feature_names, deepdrebin_importances, top_k=50):
    """
    Compare important features between TIF model and DeepDrebin model
    
    Args:
        tif_feature_names: List of top feature names from TIF model
        tif_importances: List of importance scores from TIF model
        deepdrebin_feature_names: List of top feature names from DeepDrebin model
        deepdrebin_importances: List of importance scores from DeepDrebin model
        top_k: Number of top features to compare
    
    Returns:
        comparison_dict: Dictionary with comparison statistics
    """
    # Create sets for comparison
    tif_set = set(tif_feature_names[:top_k])
    deepdrebin_set = set(deepdrebin_feature_names[:top_k])
    
    # Find common and unique features
    common_features = tif_set & deepdrebin_set
    tif_unique = tif_set - deepdrebin_set
    deepdrebin_unique = deepdrebin_set - tif_set
    
    # Create feature to importance mapping
    tif_dict = {name: score for name, score in zip(tif_feature_names, tif_importances)}
    deepdrebin_dict = {name: score for name, score in zip(deepdrebin_feature_names, deepdrebin_importances)}
    
    comparison_dict = {
        'common_features': list(common_features),
        'tif_unique': list(tif_unique),
        'deepdrebin_unique': list(deepdrebin_unique),
        'common_count': len(common_features),
        'tif_unique_count': len(tif_unique),
        'deepdrebin_unique_count': len(deepdrebin_unique),
        'tif_dict': tif_dict,
        'deepdrebin_dict': deepdrebin_dict
    }
    
    return comparison_dict


def evaluate_model_effectiveness(X_selected, y_selected, env_selected, model_path, model_type='tif'):
    """
    Evaluate model effectiveness on malware samples from a specific family.
    Note: All samples in X_selected are malware samples, so we calculate coverage/detection rate.
    
    Args:
        X_selected: Feature matrix (numpy array or sparse matrix) - all samples are malware
        y_selected: True labels (numpy array) - should all be 1 (malware)
        env_selected: Environment labels (numpy array)
        model_path: Path to the trained model
        model_type: Type of model - 'tif' or 'tif_stage1' (DrebinMLP_IRM from St1ModelTrainer), 
                    'tif_stage2' (DrebinMLP_IRM from St2ModelTrainer), or 'deepdrebin' (DrebinMLP)
    
    Returns:
        metrics_dict: Dictionary containing coverage/detection metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model based on type
    if model_type == 'tif' or model_type == 'tif_stage1':
        from stage1_trainer import St1ModelTrainer
        from model import DrebinMLP_IRM
        input_size = X_selected.shape[1]
        model = St1ModelTrainer.load_model(
            model_path=model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size,
            device=device
        )
    elif model_type == 'tif_stage2':
        from stage2_trainer import St2ModelTrainer
        from model import DrebinMLP_IRM
        input_size = X_selected.shape[1]
        model, _ = St2ModelTrainer.load_model(
            model_path=model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size
        )
        model = model.to(device)
    elif model_type == 'deepdrebin':
        from trainer import ModelTrainer
        from model import DrebinMLP
        input_size = X_selected.shape[1]
        model = ModelTrainer.load_model(
            model_path=model_path,
            model_class=DrebinMLP,
            input_size=input_size,
            device=device
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'tif' or 'deepdrebin'")
    
    model.eval()
    
    # Create dataset and dataloader
    dataset = SimpleDataset(X_selected, y_selected)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    # Collect predictions
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
    
    all_preds = np.array(all_preds)
    
    # Since all samples are malware, we can only calculate:
    # - Total malware samples
    # - How many are correctly classified as malware (TP)
    # - How many are misclassified as benign (FN)
    # - Coverage rate = TP / total_samples
    total_samples = len(all_preds)
    correctly_classified_as_malware = np.sum(all_preds == 1)  # True Positives
    misclassified_as_benign = np.sum(all_preds == 0)  # False Negatives
    
    # Calculate coverage/detection metrics
    coverage_rate = float(correctly_classified_as_malware / total_samples) if total_samples > 0 else 0.0
    miss_rate = float(misclassified_as_benign / total_samples) if total_samples > 0 else 0.0
    
    # Prediction distribution
    pred_label_dist = Counter(all_preds)
    
    metrics_dict = {
        'total_samples': total_samples,
        'correctly_classified_as_malware': int(correctly_classified_as_malware),
        'misclassified_as_benign': int(misclassified_as_benign),
        'coverage_rate': coverage_rate,
        'detection_rate': coverage_rate,  # Same as coverage rate
        'miss_rate': miss_rate,
        'pred_label_distribution': dict(pred_label_dist)
    }
    
    return metrics_dict


def print_effectiveness_report(metrics_dict, model_name, family_name):
    """
    Print a formatted report of model effectiveness on malware samples
    
    Args:
        metrics_dict: Dictionary returned from evaluate_model_effectiveness
        model_name: Name of the model (e.g., 'TIF', 'DeepDrebin')
        family_name: Name of the malware family
    """
    print("\n" + "=" * 80)
    print(f"Model Effectiveness Report: {model_name} on Family '{family_name}'")
    print("=" * 80)
    print("(Note: All samples in the dataset are malware samples)")
    
    print(f"\nSample Statistics:")
    print(f"  Total malware samples: {metrics_dict['total_samples']}")
    print(f"  Correctly classified as malware: {metrics_dict['correctly_classified_as_malware']}")
    print(f"  Misclassified as benign: {metrics_dict['misclassified_as_benign']}")
    
    print(f"\nCoverage/Detection Metrics:")
    print(f"  Coverage Rate:     {metrics_dict['coverage_rate']:.4f} ({metrics_dict['coverage_rate']:.2%})")
    print(f"  Detection Rate:    {metrics_dict['detection_rate']:.4f} ({metrics_dict['detection_rate']:.2%})")
    print(f"  Miss Rate:         {metrics_dict['miss_rate']:.4f} ({metrics_dict['miss_rate']:.2%})")
    
    print(f"\nPrediction Distribution:")
    pred_dist = metrics_dict['pred_label_distribution']
    for label, count in sorted(pred_dist.items()):
        label_name = "Benign" if label == 0 else "Malware"
        percentage = (count / metrics_dict['total_samples']) * 100 if metrics_dict['total_samples'] > 0 else 0
        print(f"  {label_name}: {count:5d} ({percentage:.2f}%)")
    
    print(f"\nInterpretation:")
    print(f"  The model correctly identifies {metrics_dict['coverage_rate']:.2%} of malware samples")
    print(f"  from the '{family_name}' family as malware.")
    if metrics_dict['miss_rate'] > 0:
        print(f"  {metrics_dict['miss_rate']:.2%} of samples are incorrectly classified as benign (missed).")


def main(select_family):
    # Configuration
    file_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features/train_data.pkl"
    tif_model_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt/stage2_model_epoch32_lr0.001_bs256.pt"
    deepdrebin_model_path = '/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt/mpc_model_epoch29_lr0.0001_bs128.pt'  # Set to your DeepDrebin model path, e.g., "/path/to/deepdrebin_model.pt"
    top_k = 20
    
    # Select data for target family
    X_selected, y_selected, env_selected, y_family_selected = select_existfamily_by_file(file_path, select_family)
    # X_selected, y_selected, env_selected, y_family_selected = select_existfamily_for_test(select_family)
    print(f"X_selected shape: {X_selected.shape}, y_selected shape: {y_selected.shape}, env_selected shape: {env_selected.shape}")
    
    # Evaluate model effectiveness
    # Handle both sparse and dense matrices
    num_samples = X_selected.shape[0] if hasattr(X_selected, 'shape') else len(X_selected)
    if num_samples > 0:
        print("\n" + "=" * 80)
        print("Evaluating Model Effectiveness")
        print("=" * 80)
        
        # Determine model type based on path
        if 'stage2' in tif_model_path.lower() or 'stg2' in tif_model_path.lower():
            tif_model_type = 'tif_stage2'
            tif_model_name = 'TIF (Stage 2)'
        elif 'stage1' in tif_model_path.lower() or 'stg1' in tif_model_path.lower():
            tif_model_type = 'tif_stage1'
            tif_model_name = 'TIF (Stage 1)'
        else:
            tif_model_type = 'tif_stage1'  # Default
            tif_model_name = 'TIF'
        
        # Evaluate TIF model
        tif_metrics = evaluate_model_effectiveness(
            X_selected, y_selected, env_selected,
            model_path=tif_model_path,
            model_type=tif_model_type
        )
        print_effectiveness_report(tif_metrics, tif_model_name, select_family[0])
        
        # Evaluate DeepDrebin model if path is provided
        if deepdrebin_model_path is not None:
            deepdrebin_metrics = evaluate_model_effectiveness(
                X_selected, y_selected, env_selected,
                model_path=deepdrebin_model_path,
                model_type='deepdrebin'
            )
            print_effectiveness_report(deepdrebin_metrics, 'DeepDrebin', select_family[0])
            
            # Compare effectiveness
            print("\n" + "=" * 80)
            print("Effectiveness Comparison")
            print("=" * 80)
            print(f"{'Metric':<30} {'TIF':<20} {'DeepDrebin':<20} {'Difference':<20}")
            print("-" * 90)
            print(f"{'Coverage Rate':<30} {tif_metrics['coverage_rate']:<20.4f} {deepdrebin_metrics['coverage_rate']:<20.4f} {tif_metrics['coverage_rate'] - deepdrebin_metrics['coverage_rate']:<20.4f}")
            print(f"{'Detection Rate':<30} {tif_metrics['detection_rate']:<20.4f} {deepdrebin_metrics['detection_rate']:<20.4f} {tif_metrics['detection_rate'] - deepdrebin_metrics['detection_rate']:<20.4f}")
            print(f"{'Miss Rate':<30} {tif_metrics['miss_rate']:<20.4f} {deepdrebin_metrics['miss_rate']:<20.4f} {tif_metrics['miss_rate'] - deepdrebin_metrics['miss_rate']:<20.4f}")
            print(f"{'Correctly Classified':<30} {tif_metrics['correctly_classified_as_malware']:<20d} {deepdrebin_metrics['correctly_classified_as_malware']:<20d} {tif_metrics['correctly_classified_as_malware'] - deepdrebin_metrics['correctly_classified_as_malware']:<20d}")
            print(f"{'Misclassified':<30} {tif_metrics['misclassified_as_benign']:<20d} {deepdrebin_metrics['misclassified_as_benign']:<20d} {tif_metrics['misclassified_as_benign'] - deepdrebin_metrics['misclassified_as_benign']:<20d}")
    else:
        print("Warning: No samples found for the selected family!")
    
    # Calculate importance for TIF model
    print("\n" + "=" * 80)
    print(f"Calculating importance for TIF model (Stage 1)...")
    print("=" * 80)
    tif_feature_names, tif_importances = calculate_importance(
        X_selected, y_selected, env_selected, 
        model_path=tif_model_path,
        feature_names_path=feature_names_path,
        top_k=top_k
    )
    
    # Output TIF results
    print(f"\nTop {top_k} Important Features for TIF model (family '{select_family[0]}'):")
    print("=" * 80)
    for i, (feature_name, importance_score) in enumerate(zip(tif_feature_names, tif_importances), 1):
        print(f"{i:3d}. {feature_name:60s} | Importance: {importance_score:.6f}")
    
    # Calculate importance for DeepDrebin model if path is provided
    if deepdrebin_model_path is not None:
        print("\n" + "=" * 80)
        print(f"Calculating importance for DeepDrebin model...")
        print("=" * 80)
        deepdrebin_feature_names, deepdrebin_importances = calculate_importance_deepdrebin(
            X_selected, y_selected, env_selected,
            model_path=deepdrebin_model_path,
            feature_names_path=feature_names_path,
            top_k=top_k
        )
        
        # Output DeepDrebin results
        print(f"\nTop {top_k} Important Features for DeepDrebin model (family '{select_family[0]}'):")
        print("=" * 80)
        for i, (feature_name, importance_score) in enumerate(zip(deepdrebin_feature_names, deepdrebin_importances), 1):
            print(f"{i:3d}. {feature_name:60s} | Importance: {importance_score:.6f}")
        
        # Compare the two models
        print("\n" + "=" * 80)
        print("Comparison between TIF and DeepDrebin models:")
        print("=" * 80)
        comparison = compare_importance(tif_feature_names, tif_importances, 
                                      deepdrebin_feature_names, deepdrebin_importances, 
                                      top_k=top_k)
        
        print(f"\nCommon features ({comparison['common_count']}/{top_k}):")
        for i, feat in enumerate(comparison['common_features'], 1):
            tif_score = comparison['tif_dict'].get(feat, 0)
            dd_score = comparison['deepdrebin_dict'].get(feat, 0)
            print(f"  {i:3d}. {feat:60s} | TIF: {tif_score:.6f} | DeepDrebin: {dd_score:.6f}")
        
        print(f"\nTIF unique features ({comparison['tif_unique_count']}):")
        for i, feat in enumerate(list(comparison['tif_unique'])[:10], 1):  # Show top 10
            score = comparison['tif_dict'].get(feat, 0)
            print(f"  {i:3d}. {feat:60s} | Importance: {score:.6f}")
        
        print(f"\nDeepDrebin unique features ({comparison['deepdrebin_unique_count']}):")
        for i, feat in enumerate(list(comparison['deepdrebin_unique'])[:10], 1):  # Show top 10
            score = comparison['deepdrebin_dict'].get(feat, 0)
            print(f"  {i:3d}. {feat:60s} | Importance: {score:.6f}")
    else:
        print("\nNote: Set 'deepdrebin_model_path' to compare with DeepDrebin model")

if __name__ == "__main__":
    # 1) Compute unseen families (present in test but not in train)
    X_test_all, y_test_all, env_test_all, y_family_test_all = get_all_test_data()
    family_type_test = np.unique(y_family_test_all)
    print(
        f"All test data shapes: X={X_test_all.shape}, y={y_test_all.shape}, "
        f"env={env_test_all.shape}, y_family={y_family_test_all.shape}"
    )

    with open(train_data_path, 'rb') as f:
        data_train = pickle.load(f)
    X_train = data_train['X']
    y_train = data_train['y']
    y_family_train = data_train['y_family']
    env_train = data_train['env']
    family_type_train = np.unique(y_family_train)
    print(
        f"Train data shapes: X={X_train.shape}, y={y_train.shape}, "
        f"env={env_train.shape}, y_family={y_family_train.shape}"
    )

    diff_test_family = list(set(family_type_test) - set(family_type_train))
    print(f"Unseen test family ids (diff_test_family): {diff_test_family}")

    # 2) Draw UMAP for training vs unseen families for both TIF and DeepDrebin
    tif_model_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt/stage2_model_epoch32_lr0.001_bs256.pt"
    deep_model_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt/mpc_model_epoch29_lr0.0001_bs128.pt"

    # TIF (Stage 2)
    reps_train_tif, reps_unseen_tif = get_representations_train_vs_unseen(
        model_type="tif_stage2",
        model_path=tif_model_path,
        X_train=X_train,
        y_train=y_train,
        y_family_train=y_family_train,
        X_test=X_test_all,
        y_test=y_test_all,
        y_family_test=y_family_test_all,
        unseen_family_ids=diff_test_family,
        train_exclude_family=37,
        sample_size=2000,
    )
    plot_2d_embeddings_two_groups(
        reps_train_tif,
        reps_unseen_tif,
        model_name="TIF (Stage 2)",
        method="umap",
        save_path=os.path.join(figures_folder, "tif_umap_train_vs_unseen.png"),
    )

    # DeepDrebin
    reps_train_dd, reps_unseen_dd = get_representations_train_vs_unseen(
        model_type="deepdrebin",
        model_path=deep_model_path,
        X_train=X_train,
        y_train=y_train,
        y_family_train=y_family_train,
        X_test=X_test_all,
        y_test=y_test_all,
        y_family_test=y_family_test_all,
        unseen_family_ids=diff_test_family,
        train_exclude_family=37,
        sample_size=2000,
    )
    plot_2d_embeddings_two_groups(
        reps_train_dd,
        reps_unseen_dd,
        model_name="DeepDrebin",
        method="umap",
        save_path=os.path.join(figures_folder, "deepdrebin_umap_train_vs_unseen.png"),
    )

    # # 1) TIF (stage 2)
    # reps_tr_air_tif, reps_te_air_tif, reps_te_hid_tif = get_family_representations_for_model(
    #     model_type="tif_stage2",
    #     model_path=tif_model_path,
    # )
    # plot_2d_embeddings(
    #     reps_tr_air_tif,
    #     reps_te_air_tif,
    #     reps_te_hid_tif,
    #     model_name="TIF (Stage 2)",
    #     method="umap",
    #     save_path=os.path.join(figures_folder, "tif_umap.png")
    # )
 
    # # 2) DeepDrebin
    # reps_tr_air_dd, reps_te_air_dd, reps_te_hid_dd = get_family_representations_for_model(
    #     model_type="deepdrebin",
    #     model_path=deep_model_path,
    # )
    # plot_2d_embeddings(
    #     reps_tr_air_dd,
    #     reps_te_air_dd,
    #     reps_te_hid_dd,
    #     model_name="DeepDrebin",
    #     method="umap",
    #     save_path=os.path.join(figures_folder, "deepdrebin_umap.png")
    # )

