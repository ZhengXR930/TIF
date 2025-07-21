import sys
sys.path.append('/cs/academic/phd3/xinrzhen/xinran/SaTML')
import torch
import torch.nn as nn
import numpy as np
import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from model import DrebinMLP
# set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from loss import PALSoftWithInterMargin
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

# Set random seeds for reproducibility
def set_seed(seed=42):
    print("seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def generate_months(start_date, end_date):
    start_year, start_month = map(int, start_date.split('-'))
    end_year, end_month = map(int, end_date.split('-'))

    months = []
    for year in range(start_year, end_year + 1):
        if year == start_year:
            start = start_month
        else:
            start = 1
        
        if year == end_year:
            end = end_month
        else:
            end = 12
        
        for month in range(start, end + 1):
            formatted_month = f"{year}-{month:02d}"
            months.append(formatted_month)
    
    return months

def load_data(train_path):
    train_data = np.load(train_path, allow_pickle=True)
    X = train_data['X_train']
    y = train_data['y_train']
    y_mal_family = train_data['y_mal_family']
    ben_len = X.shape[0] - y_mal_family.shape[0]
    y_ben_family = np.full(ben_len, 'benign')
    y_family = np.concatenate((y_mal_family, y_ben_family), axis=0)
    y_binary = np.array([1 if item != 0 else 0 for item in y])
    return X, y_binary, y_family

def train_nn(model, X_train, y_train, check_point, lr=0.0001, epochs=30, batch_size=256, best_model_name="best_model.pth"):
    model.to(device)
    model.train()
    best_f1 = 0
    criterion_bn = nn.CrossEntropyLoss()
    criterion_con = PALSoftWithInterMargin(
        device=device,
        input_dim=model.emb_dim,
        embed_dim=128,
        num_classes=2,
        n_proxy=5,
        tau=0.2,
        margin=1.5,
        lambda_margin=0.05,
        lambda_div=0.01
    ).to(device)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).long().to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion_con.parameters()), lr=lr
    )

    for epoch in range(epochs):
        all_preds = []
        all_labels = []
        total_loss = 0.0
        step = 0
        interval = 20
        for xb, yb in loader:
            optimizer.zero_grad()
            logits, feature = model(xb)
            logits = logits.squeeze(-1)
            loss_bn = criterion_bn(logits, yb)
            loss_con = criterion_con(feature, yb)
            loss = loss_bn + loss_con
            # loss = loss_bn
            # print(f"Loss: {loss.item()}, Loss_bn: {loss_bn.item()}, Loss_con: {loss_con.item()}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            all_preds.append(logits.detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())
            if step % interval == 0:
                print(f"Step {step}, Loss: {loss.item()}, Loss_bn: {loss_bn.item()}, Loss_con: {loss_con.item()}")
            step += 1
        
        # Use the correct way to get the number of samples
        avg_loss = total_loss / X_train.shape[0]
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        pred_classes = np.argmax(all_preds, axis=1)
        acc = accuracy_score(all_labels, pred_classes)
        precision = precision_score(all_labels, pred_classes, average='binary')
        recall = recall_score(all_labels, pred_classes, average='binary')
        f1 = f1_score(all_labels, pred_classes, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(check_point, best_model_name))
            print(f"Best F1: {f1:.4f}, saved at {os.path.join(check_point, best_model_name)}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return model



def evaluate_nn(model, X_test, y_test,check_point=None):
    model.to(device)
    if check_point is not None:
        model.load_state_dict(torch.load(check_point, map_location=device, weights_only=True))
        print(f"Loaded model from {check_point}")
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(X_test).float().to(device))
        logits_np = logits.detach().cpu().numpy()
        preds = np.argmax(logits_np, axis=1)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='binary')
        recall = recall_score(y_test, preds, average='binary')
        f1 = f1_score(y_test, preds, average='binary')
    print(f"NN Test Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return acc, precision, recall, f1

def evaluate_monthly(model, start_date, end_date,model_save_path=None):
    month_list = generate_months(start_date, end_date)
    for month in month_list:
        print(f"Evaluating {month}")
        train_path = f"/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/data/gen_apigraph_drebin/{month}_selected.npz"
        X, y_binary, y_family = load_data(train_path)
        evaluate_nn(model, X, y_binary, model_save_path)

def extract_features(model, dataloader):
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            _, feat = model(xb)  
            all_feats.append(feat.cpu())
            all_labels.append(yb.cpu())
    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    print(f"Features shape: {all_feats.shape}, Labels shape: {all_labels.shape}")
    return all_feats, all_labels

def sample_features(features, labels, num_samples=5000, seed =42):
    np.random.seed(seed)
    features = np.array(features)
    labels = np.array(labels)

    unique_classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    sampled_features = []
    sampled_labels = []
    
    for cls, count in zip(unique_classes, counts):
        cls_mask = (labels == cls)
        cls_features = features[cls_mask]
        cls_labels = labels[cls_mask]
        
        proportion = count / total
        n_samples = max(1, int(proportion * num_samples)) 
        
        if len(cls_features) <= n_samples:
            sampled_features.append(cls_features)
            sampled_labels.append(cls_labels)
        else:
            idx = np.random.choice(len(cls_features), n_samples, replace=False)
            sampled_features.append(cls_features[idx])
            sampled_labels.append(cls_labels[idx])
    
    sampled_features = np.concatenate(sampled_features, axis=0)
    sampled_labels = np.concatenate(sampled_labels, axis=0)
    print(f"Sampled shape: {sampled_features.shape}, {sampled_labels.shape}")
    return sampled_features, sampled_labels

def plot_features(features, labels, title='t-SNE of Features'):
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(features)
    
    plt.figure(figsize=(6,5))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap='coolwarm', s=5)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(f"/cs/academic/phd3/xinrzhen/xinran/retry/fig/{title}.png")


if __name__ == "__main__":

    check_point = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt"
    train_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/data/gen_apigraph_drebin/2012-01to2012-12_selected.npz"

    # Load and split data
    X, y_binary, y_family = load_data(train_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

    # Define model, optimizer, loss
    best_model_name = "best_model_con.pth"
    model = DrebinMLP(input_size=X.shape[1])
    model_save_path = os.path.join(check_point, best_model_name)
    # model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.to(device)

    # dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).long().to(device))
    # loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    # features, labels = extract_features(model, loader)
    # sampled_features, sampled_labels = sample_features(features, labels, num_samples=2000, seed =42)
    # plot_features(sampled_features, sampled_labels, title='learned features bce')

    # Train
    train_nn(model, X_train, y_train, check_point, lr=0.0001, epochs=60, batch_size=256, best_model_name=best_model_name)

    # Evaluate
    evaluate_nn(model, X_test, y_test, model_save_path)
    # evaluate_monthly(model, "2013-01", "2013-12", model_save_path)

    