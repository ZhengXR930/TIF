
import sys
sys.path.append('/cs/academic/phd3/xinrzhen/xinran/SaTML')
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,confusion_matrix
import logging
import numpy as np
import tesseract.metrics as tm
from scipy.sparse import vstack, csr_matrix
from sklearn import svm
from collections import Counter
from collections import defaultdict
import os
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
from trainer import ModelTrainer, CustomDataset
from stage1_trainer import Stg1CustomDataset, St1ModelTrainer
from stage2_trainer import Stg2CustomDataset, St2ModelTrainer
from model import DrebinMLP, DrebinMLP_IRM
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import base_line.drebin as drebin
import utils
import t_stability as ts
import torch
import random

data_folder = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features/"
result_folder = "/cs/academic/phd3/xinrzhen/xinran/SaTML/result_rf"
save_folder = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt"

# set seed
os.environ["PYTHONHASHSEED"] = "1"
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
def eval_svm(train_path, val_path, test_list):
    x_train, y_train, _, _ = utils.load_train_overall(train_path)
    x_val, y_val, _, _ = utils.load_train_overall(val_path)
    clf = drebin.drebin_svm_train(x_train, y_train)
    _, _, f1 = drebin.drebin_svm_pred(clf, x_val, y_val)

    result_f1 = []
    result_f1.append(f1)
    drebin.drebin_svm_monthly(clf,result_f1, result_folder, data_folder, test_list, f'svm_test_file.csv')


def eval_t_stability(test_list):
    ts.creat_t_stability()
    ts_path = "/root/malware/ELSA/checkpoints/t_stability.pkl"
    w, b, f1 = ts.retrain_svm(ts_path)

    i = 1
    results_f1 = []
    results_f1.append(f1)
    with open(os.path.join(result_folder, f'ts_test_file_{i}.csv'), 'w') as f:
        f.write("month,precision,recall,f1\n")
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        x_test = x_test.toarray()
        y_pred = ts.svm_cb_predict(x_test, w, b)
        precision = precision_score(y_test, y_pred,average='macro')
        recall = recall_score(y_test, y_pred,average='macro')
        f1 = f1_score(y_test, y_pred,average='macro')
        print(f"test month: {month}, test metrics: {precision}, {recall}, {f1}")
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(os.path.join(result_folder, f'ts_test_file_{i}.csv'), 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")


def eval_deepdrebin(train_path,test_list,best_model_path=None):
    x_train, y_train, _, _ = utils.load_train_overall(train_path)
    x_val, y_val, _, _ = utils.load_train_overall(val_path)
    # transfer to dense matrix
    x_train = csr_matrix(x_train).todense()
    x_val = csr_matrix(x_val).todense()
    input_size= x_train.shape[1]

    if best_model_path is None:
        model = DrebinMLP(input_size=input_size)
        trainer = ModelTrainer(
        model=model,
        device='cuda',
        batch_size=128,
        learning_rate=0.0001,
        save_dir=save_folder)
        
        best_model_path = trainer.train(x_train, x_val, y_train, y_val, epochs=30)
        print(f"best model path: {best_model_path}")
    else:
        model = ModelTrainer.load_model(
        model_path=best_model_path,
        model_class=DrebinMLP,
        input_size=input_size
        )

        trainer = ModelTrainer(
        model=model,
        device='cuda',
        batch_size=128,
        learning_rate=0.0001,
        save_dir=save_folder)

    val_dataset = CustomDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)
    f1_val = val_metrics['f1']

    monthly_results_path = os.path.join(result_folder, f'deep_test_file.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1\n")
    results_f1 = []
    results_f1.append(f1_val)
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}")



def eval_mpc_stage_1(train_path, test_list, best_model_path=None):
    x_train, y_train, env_train, t_train = utils.load_train_overall(train_path)
    x_val, y_val, env_val, t_val = utils.load_train_overall(val_path)
    input_size = x_train.shape[1]

    if best_model_path is None:
        model = DrebinMLP_IRM(input_size=input_size)
        trainer = St1ModelTrainer(
            model=model,
            device='cuda',
            batch_size=256,
            learning_rate=0.0001,
            con_loss_weight=1.0,
            save_dir=save_folder)

        best_model_path = trainer.train(x_train, x_val, y_train, y_val, env_train, env_val, epochs=60)
        print(f"best model path: {best_model_path}")
    
    else:
        print(f"load best model from {best_model_path}")
        model = St1ModelTrainer.load_model(
        model_path=best_model_path,
        model_class=DrebinMLP_IRM,
        input_size=input_size
        )
        trainer = St1ModelTrainer(
            model=model,
            device='cuda',
            batch_size=1024,
            learning_rate=0.0005,
            con_loss_weight=1.0,
            save_dir=save_folder)
    
    val_dataset = Stg1CustomDataset(x_val, y_val, env_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)
    f1_val = val_metrics['f1']

    monthly_results_path = os.path.join(result_folder, f'stage1_test_file.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1\n")
    results_f1 = []
    results_f1.append(f1_val)
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}")


def eval_mpc_stage_2(train_path, test_list, best_model_path=None):
    x_train, y_train, env_train, t_train = utils.load_train_overall(train_path)
    x_val, y_val, env_val, t_val = utils.load_train_overall(val_path)
    input_size = x_train.shape[1]

    if best_model_path is None:
        model = DrebinMLP_IRM(input_size=input_size)
        trainer = St2ModelTrainer(
            model=model,
            device='cuda',
            batch_size=256,
            learning_rate=0.0001,
            con_loss_weight=1.0,
            penalty_weight=0.01,
            save_dir=save_folder)

        best_model_path = trainer.train(x_train, x_val, y_train, y_val, env_train, env_val, epochs=60)
        print(f"best model path: {best_model_path}")
    
    else:
        print(f"load best model from {best_model_path}")
        model, custom_loss_state_dict = St2ModelTrainer.load_model(
            model_path=best_model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size
        )
        trainer = St2ModelTrainer(
            model=model,
            device='cuda',
            batch_size=256,
            learning_rate=0.0001,
            con_loss_weight=1.0,
            penalty_weight=0.01,
            save_dir=save_folder,
            custom_loss_state_dict=custom_loss_state_dict
        )
    
    val_dataset = Stg2CustomDataset(x_val, y_val, env_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)
    f1_val = val_metrics['f1']


    monthly_results_path = os.path.join(result_folder, f'irm_test_file.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1\n")
    results_f1 = []
    results_f1.append(f1_val)
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}")


def eval_tif(train_path, test_list, best_stg1_model_path, best_stg2_model_path=None):
    x_train, y_train, env_train, t_train = utils.load_train_overall(train_path)
    x_val, y_val, env_val, t_val = utils.load_train_overall(val_path)

    input_size = x_train.shape[1]

    if best_stg1_model_path is None:
        print("train stage 1 model doesn't exist!!")
    
    elif best_stg2_model_path is None:
        print(f"load best stg1 model from {best_stg1_model_path}")
        # Load both model weights and MPC proxy parameters from stage 1 checkpoint
        model, custom_loss_state_dict = St2ModelTrainer.load_model(
            model_path=best_stg1_model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size
        )
        trainer = St2ModelTrainer(
            model=model,
            device='cuda',
            batch_size=256,
            learning_rate=0.001,
            con_loss_weight=1.0,
            penalty_weight=0.05,
            save_dir=save_folder,
            custom_loss_state_dict=custom_loss_state_dict
        )

        trainer.reset_optimizer(learning_rate=0.0001)
        best_model_path = trainer.train(x_train, x_val, y_train, y_val, env_train, env_val, epochs=50)
        print(f"best model path: {best_model_path}")
    
    elif best_stg2_model_path is not None:

        print(f"load best stg2 model from {best_stg2_model_path}")
        model, custom_loss_state_dict = St2ModelTrainer.load_model(
            model_path=best_stg2_model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size
        )
        trainer = St2ModelTrainer(
            model=model,
            device='cuda',
            batch_size=256,
            learning_rate=0.0001,
            con_loss_weight=1.0,
            penalty_weight=0.01,
            save_dir=save_folder,
            custom_loss_state_dict=custom_loss_state_dict
        )

    val_dataset = Stg2CustomDataset(x_val, y_val, env_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)
    f1_val = val_metrics['f1']


    monthly_results_path = os.path.join(result_folder, f'tif_test_file.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1\n")
    results_f1 = []
    results_f1.append(f1_val)
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}")


def eval_mpc(train_path,test_list,best_model_path=None):
    x_train, y_train, _, _ = utils.load_train_overall(train_path)
    x_val, y_val, _, _ = utils.load_train_overall(val_path)
    # transfer to dense matrix
    x_train = csr_matrix(x_train).todense()
    x_val = csr_matrix(x_val).todense()
    input_size= x_train.shape[1]

    if best_model_path is None:
        model = DrebinMLP_IRM(input_size=input_size)
        trainer = ModelTrainer(
        model=model,
        device='cuda',
        batch_size=256,
        learning_rate=0.0001,
        con_loss_weight=1.0,
        save_dir=save_folder)
        
        best_model_path = trainer.train(x_train, x_val, y_train, y_val, epochs=60)
        print(f"best model path: {best_model_path}")
    else:
        model = ModelTrainer.load_model(
        model_path=best_model_path,
        model_class=DrebinMLP_IRM,
        input_size=input_size
        )

        trainer = ModelTrainer(
        model=model,
        device='cuda',
        batch_size=128,
        learning_rate=0.0001,
        save_dir=save_folder)

    val_dataset = CustomDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)
    f1_val = val_metrics['f1']

    monthly_results_path = os.path.join(result_folder, f'mpc_test_file.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1\n")
    results_f1 = []
    results_f1.append(f1_val)
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}")   

def active_learning(trained_model_path, train_data_path, val_data_path, test_list, result_folder, budget=100):

    x_train, y_train, env_train, t_train = utils.load_train_overall(train_data_path)
    x_val, y_val, env_val, t_val = utils.load_train_overall(val_data_path)
    input_size = x_train.shape[1]

    x_train = csr_matrix(x_train).todense()
    y_train = csr_matrix(y_train).todense()
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to('cuda')
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to('cuda').squeeze()
    env_train_tensor = torch.tensor(env_train, dtype=torch.long).to('cuda').squeeze()

    

    monthly_results_path = os.path.join(result_folder, f'active_learning_test_file.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1,update\n")

    best_model_path = trained_model_path
    for month in test_list:
        update_flag = 1
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        env_test = np.full(len(y_test), 3, dtype=int)
        
        x_test = csr_matrix(x_test).todense()
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to('cuda')
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to('cuda').squeeze()
        env_test_tensor = torch.tensor(env_test, dtype=torch.long).to('cuda').squeeze()

        print(f"load model from {best_model_path}")

        model = St1ModelTrainer.load_model(
            model_path=best_model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size
            )
        trainer = St1ModelTrainer(
            model=model,
            device='cuda',
            batch_size=256,
            learning_rate=0.0001,
            con_loss_weight=1.0,
            save_dir=save_folder)

        test_dataset = Stg1CustomDataset(x_test, y_test, env_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        
        if f1 > 0.9:
            print(f"month: {month}, f1 score is already high enough, skip active learning")
            update_flag = 0
            with open(monthly_results_path, 'a') as f:
                f.write(f"{month},{precision},{recall},{f1},{update_flag}\n")
            continue

        outputs, _ = model(x_test_tensor)
        uncertainty = 1.0 - torch.max(outputs, dim=1).values
        indices = torch.argsort(uncertainty, descending=True)[:budget]
        # print(f"month: {month}, uncertainty: {uncertainty[indices]}")

        x_test_tensor_selected = x_test_tensor[indices]
        y_test_tensor_selected = y_test_tensor[indices]
        env_test_tensor_selected = env_test_tensor[indices]

        x_combined = torch.cat((x_train_tensor, x_test_tensor_selected), dim=0)
        y_combined = torch.cat((y_train_tensor, y_test_tensor_selected), dim=0)
        env_combined = torch.cat((env_train_tensor, env_test_tensor_selected), dim=0)

        x_train_tensor = x_combined
        y_train_tensor = y_combined
        env_train_tensor = env_combined

        x_train_np = x_train_tensor.cpu().numpy()
        y_train_np = y_train_tensor.cpu().numpy()
        env_train_np = env_train_tensor.cpu().numpy()
        print(f"env distribution: {np.unique(env_train, return_counts=True)}")
        print(f"env distribution: {np.unique(env_train_np, return_counts=True)}")

        x_train_np_new, x_val_np, y_train_np_new, y_val_np, env_train_np_new, env_val_np = train_test_split(
            x_train_np, y_train_np, env_train_np,
            test_size=0.2,
            random_state=42,
            stratify=y_train_np
        )
        
        print(f"original train data shape: {y_train_tensor.shape}, new train data shape: {y_test_tensor_selected.shape}")
        print(f"original test data shape: {x_test_tensor.shape}, selected test data shape: {x_test_tensor_selected.shape}")
        print(f"original train data shape: {y_train_tensor.shape}, new train data shape: {y_combined.shape}")
        print(f"env distribution: {np.unique(env_train_np, return_counts=True)}")

        # break

        best_model_path = trainer.train(x_train_np_new, x_val_np, y_train_np_new, y_val_np, env_train_np_new, env_val_np, epochs=30)
        print(f"best model path: {best_model_path}")

        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{update_flag}\n")

def active_learning_probability(trained_model_path, train_data_path, val_data_path, test_list, result_folder, budget=100):

    x_train, y_train, env_train, t_train = utils.load_train_overall(train_data_path)
    x_val, y_val, env_val, t_val = utils.load_train_overall(val_data_path)
    input_size = x_train.shape[1]

    x_train = csr_matrix(x_train).todense()
    y_train = csr_matrix(y_train).todense()
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to('cuda')
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to('cuda').squeeze()
    env_train_tensor = torch.tensor(env_train, dtype=torch.long).to('cuda').squeeze()

    

    monthly_results_path = os.path.join(result_folder, f'active_learning_test_file.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1,high_mean,tau_abs,high_ratio,high_count,update\n")

    best_model_path = trained_model_path
    for month in test_list:
        update_flag = 1
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        env_test = np.full(len(y_test), 3, dtype=int)
        
        x_test = csr_matrix(x_test).todense()
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to('cuda')
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to('cuda').squeeze()
        env_test_tensor = torch.tensor(env_test, dtype=torch.long).to('cuda').squeeze()

        print(f"load model from {best_model_path}")

        model = St1ModelTrainer.load_model(
            model_path=best_model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size
            )
        trainer = St1ModelTrainer(
            model=model,
            device='cuda',
            batch_size=256,
            learning_rate=0.0001,
            con_loss_weight=1.0,
            save_dir=save_folder)

        test_dataset = Stg1CustomDataset(x_test, y_test, env_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']


        model.eval()
        with torch.no_grad():
            outputs, _ = model(x_test_tensor)
            probs = torch.softmax(outputs, dim=1)
            uncertainty = 1.0 - torch.max(probs, dim=1).values
            indices = torch.argsort(uncertainty, descending=True)[:budget]

        N = len(uncertainty)
        tau_abs = 0.3
        min_ratio = 0.05
        min_count = max(3*budget, 200)
        print(f"month: {month}, uncertainty: {torch.max(uncertainty).item()}, {torch.min(uncertainty).item()}")
        high_mask = uncertainty >= tau_abs
        high_count = int(high_mask.sum().item())
        high_ratio = high_count / max(N, 1)
        high_mean  = float(uncertainty[high_mask].mean().item()) if high_count > 0 else 0.0
        print(f"month: {month}, high_ratio: {high_ratio:.4f}, high_count: {high_count}, high_mean: {high_mean:.4f}")
        need_update = (high_ratio >= min_ratio) and (high_count >= min_count)
        

        if not need_update:
            update_flag = 0
            with open(monthly_results_path, 'a') as f:
                f.write(f"{month},{precision},{recall},{f1},{high_mean:.4f},{tau_abs:.4f},{high_ratio:.4f},{high_count},{update_flag}\n")
            continue


        x_test_tensor_selected = x_test_tensor[indices]
        y_test_tensor_selected = y_test_tensor[indices]
        env_test_tensor_selected = env_test_tensor[indices]

        x_combined = torch.cat((x_train_tensor, x_test_tensor_selected), dim=0)
        y_combined = torch.cat((y_train_tensor, y_test_tensor_selected), dim=0)
        env_combined = torch.cat((env_train_tensor, env_test_tensor_selected), dim=0)

        x_train_tensor = x_combined
        y_train_tensor = y_combined
        env_train_tensor = env_combined

        x_train_np = x_train_tensor.cpu().numpy()
        y_train_np = y_train_tensor.cpu().numpy()
        env_train_np = env_train_tensor.cpu().numpy()
        print(f"env distribution: {np.unique(env_train, return_counts=True)}")
        print(f"env distribution: {np.unique(env_train_np, return_counts=True)}")

        x_train_np_new, x_val_np, y_train_np_new, y_val_np, env_train_np_new, env_val_np = train_test_split(
            x_train_np, y_train_np, env_train_np,
            test_size=0.2,
            random_state=42,
            stratify=y_train_np
        )
        
        print(f"original train data shape: {y_train_tensor.shape}, new train data shape: {y_test_tensor_selected.shape}")
        print(f"original test data shape: {x_test_tensor.shape}, selected test data shape: {x_test_tensor_selected.shape}")
        print(f"original train data shape: {y_train_tensor.shape}, new train data shape: {y_combined.shape}")
        print(f"env distribution: {np.unique(env_train_np, return_counts=True)}")

        # break

        best_model_path = trainer.train(x_train_np_new, x_val_np, y_train_np_new, y_val_np, env_train_np_new, env_val_np, epochs=30)
        print(f"best model path: {best_model_path}")

        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{update_flag}\n")


if __name__ == "__main__":

    train_path = os.path.join(data_folder, "train_data.pkl")
    val_path = os.path.join(data_folder, "val_data.pkl")
    test_list = ['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
                '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12',
                '2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
                '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12',
                '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
                '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12',
                '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
                '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
                '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11']

    """
    train base line models
    """
    # drebin
    # eval_svm(train_path, val_path, test_list)

    # deepdrebin
    # eval_deepdrebin(train_path, test_list, best_model_path=None)

    # t-stability
    # eval_t_stability()

    # mpc
    # eval_mpc(train_path, test_list, best_model_path=None)


    # tif:mpc stage 1
    # best_model_path = None
    # eval_mpc_stage_1(train_path, test_list, best_model_path=best_model_path)

    # tif:mpc stage 2
    # best_model_path = None
    # eval_mpc_stage_2(train_path, test_list, best_model_path=best_model_path)

    # tif:overall (load stg1 first - reset optimizer - train stg2)
    best_stg1_model_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt/stage1_model_epoch59_lr0.0001_bs256.pt"
    best_stg2_model_path = None
    eval_tif(train_path, test_list, best_stg1_model_path, best_stg2_model_path)
    # trained_model_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt/stage1_model_epoch59_lr0.0001_bs256.pt"
    # train_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features/train_data.pkl"
    # val_path = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features/val_data.pkl"
    # test_list = ['2015-01','2015-02','2015-03','2015-04','2015-05','2015-06','2015-07','2015-08','2015-09','2015-10','2015-11','2015-12',
    #             '2016-01','2016-02','2016-03','2016-04','2016-05','2016-06','2016-07','2016-08','2016-09','2016-10','2016-11','2016-12',
    #             '2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12',
    #             '2018-01','2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12',
    #             '2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12',
    #             '2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12',
    #             '2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12',
    #             ]
    # # active_learning(trained_model_path, train_path, val_path, test_list, result_folder)
    # active_learning_probability(trained_model_path, train_path, val_path, test_list, result_folder)



    