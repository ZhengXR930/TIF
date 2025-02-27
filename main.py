
import sys
sys.path.append('/root/malware/ELSA')
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
from stage1_trainer import Stg1CustomDataset, St1ModelTrainer, BalancedUniformEnvSampler
from stage2_trainer import Stg2CustomDataset, St2ModelTrainer
from model import DrebinMLP
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import base_line.drebin as drebin
import utils
import t_stability as ts


data_folder = "/root/malware/processed_features"
result_folder = "/root/malware/ELSA/results"
save_folder = "/root/malware/ELSA/checkpoints"

test_list_1 = ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12']
test_list_2 = ['2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06']
test_list_3 = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12']
test_list_4 = ['2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06']



    
def eval_svm(train_path):
    x_train, x_val, y_train, y_val = utils.load_train_data(train_path)
    clf = drebin.drebin_svm_train(x_train, y_train)
    _, _, f1 = drebin.drebin_svm_pred(clf, x_val, y_val)

    i = 1
    for test_list in [test_list_1, test_list_2, test_list_3, test_list_4]:
        result_f1 = []
        result_f1.append(f1)
        drebin.drebin_svm_monthly(clf, result_f1, result_folder, data_folder, test_list, f'svm_test_file_{i}')
        i += 1

def eval_t_stability():
    ts.creat_t_stability()
    ts_path = "/root/malware/ELSA/checkpoints/t_stability.pkl"
    w, b, f1 = ts.retrain_svm(ts_path)

    i = 1
    for test_list in [test_list_1, test_list_2, test_list_3, test_list_4]:
        results_f1 = []
        results_f1.append(f1)
        with open(os.path.join(result_folder, f'ts_test_file_{i}.csv'), 'w') as f:
            f.write("month,precision,recall,f1\n")
        for month in test_list:
            number = str(i)
            file_path = os.path.join(data_folder, f"test_features_round_{number}" ,f"{month}.pkl")
            x_test, y_test = utils.load_single_month_data(file_path)
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
        i += 1


def eval_deepdrebin(train_path,best_model_path=None):
    x_train, x_val, y_train, y_val = utils.load_train_data(train_path)
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

    i = 1
    for test_list in [test_list_1, test_list_2, test_list_3, test_list_4]:
        monthly_results_path = os.path.join(result_folder, f'deep_test_file_{i}.csv')
        with open(monthly_results_path, 'w') as f:
            f.write("month,precision,recall,f1\n")
        results_f1 = []
        results_f1.append(f1_val)
        for month in test_list:
            number = str(i)
            file_path = os.path.join(data_folder, f"test_features_round_{number}" ,f"{month}.pkl")
            x_test, y_test = utils.load_single_month_data(file_path)
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
        i += 1


def eval_mpc_stage_1(train_path, best_model_path=None):
    x_train, x_val, y_train, y_val, env_train, env_val = utils.get_train_dataset_envs(train_path, type='year')
    input_size = x_train.shape[1]

    if best_model_path is None:
        model = DrebinMLP(input_size=input_size)
        trainer = St1ModelTrainer(
            model=model,
            device='cuda',
            batch_size=1024,
            learning_rate=0.0005,
            save_dir=save_folder)

        best_model_path = trainer.train(x_train, x_val, y_train, y_val, env_train, env_val, epochs=50)
        print(f"best model path: {best_model_path}")
    
    else:
        print(f"load best model from {best_model_path}")
        model = St1ModelTrainer.load_model(
        model_path=best_model_path,
        model_class=DrebinMLP,
        input_size=input_size
        )
        trainer = St1ModelTrainer(
            model=model,
            device='cuda',
            batch_size=1024,
            learning_rate=0.0005,
            save_dir=save_folder)
    
    val_dataset = Stg1CustomDataset(x_val, y_val, env_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)
    f1_val = val_metrics['f1']

    i = 1
    for test_list in [test_list_1, test_list_2, test_list_3, test_list_4]:
        monthly_results_path = os.path.join(result_folder, f'mpc_test_file_{i}.csv')
        with open(monthly_results_path, 'w') as f:
            f.write("month,precision,recall,f1\n")
        results_f1 = []
        results_f1.append(f1_val)
        for month in test_list:
            number = str(i)
            file_path = os.path.join(data_folder, f"test_features_round_{number}" ,f"{month}.pkl")
            x_test, y_test = utils.load_single_month_data(file_path)
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
        i += 1

def eval_mpc_stage_2(train_path, best_model_path=None):
    x_train, x_val, y_train, y_val, env_train, env_val = utils.get_train_dataset_envs(train_path, type='year')
    input_size = x_train.shape[1]

    if best_model_path is None:
        model = DrebinMLP(input_size=input_size)
        trainer = St2ModelTrainer(
            model=model,
            device='cuda',
            batch_size=1024,
            learning_rate=0.0005,
            save_dir=save_folder)

        best_model_path = trainer.train(x_train, x_val, y_train, y_val, env_train, env_val, epochs=50)
        print(f"best model path: {best_model_path}")
    
    else:
        print(f"load best model from {best_model_path}")
        model = St2ModelTrainer.load_model(
        model_path=best_model_path,
        model_class=DrebinMLP,
        input_size=input_size
        )
        trainer = St2ModelTrainer(
            model=model,
            device='cuda',
            batch_size=1024,
            learning_rate=0.0005,
            save_dir=save_folder)
    
    val_dataset = Stg2CustomDataset(x_val, y_val, env_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)
    f1_val = val_metrics['f1']

    i = 1
    for test_list in [test_list_1, test_list_2, test_list_3, test_list_4]:
        monthly_results_path = os.path.join(result_folder, f'irm_test_file_{i}.csv')
        with open(monthly_results_path, 'w') as f:
            f.write("month,precision,recall,f1\n")
        results_f1 = []
        results_f1.append(f1_val)
        for month in test_list:
            number = str(i)
            file_path = os.path.join(data_folder, f"test_features_round_{number}" ,f"{month}.pkl")
            x_test, y_test = utils.load_single_month_data(file_path)
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
        i += 1

def eval_tif(train_path, best_stg1_model_path, best_stg2_model_path=None):
    x_train, x_val, y_train, y_val, env_train, env_val = utils.get_train_dataset_envs(train_path, type='year')

    input_size = x_train.shape[1]

    if best_stg1_model_path is None:
        print("train stage 1 model doesn't exist!!")
    
    elif best_stg2_model_path is None:
        print(f"load best stg1 model from {best_stg1_model_path}")
        model = St2ModelTrainer.load_model(
        model_path=best_stg1_model_path,
        model_class=DrebinMLP,
        input_size=input_size
        )
        trainer = St2ModelTrainer(
            model=model,
            device='cuda',
            batch_size=1024,
            learning_rate=0.0005,
            save_dir=save_folder)

        trainer.reset_optimizer(learning_rate=0.001)
        trainer.penalty_weight = 10000.0
        best_model_path = trainer.train(x_train, x_val, y_train, y_val, env_train, env_val, epochs=30)
    
    elif best_stg2_model_path is not None:

        print(f"load best stg2 model from {best_stg2_model_path}")
        model = St2ModelTrainer.load_model(
        model_path=best_stg2_model_path,
        model_class=DrebinMLP,
        input_size=input_size
        )
        trainer = St2ModelTrainer(
            model=model,
            device='cuda',
            batch_size=1024,
            learning_rate=0.0005,
            save_dir=save_folder)

    val_dataset = Stg2CustomDataset(x_val, y_val, env_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)
    f1_val = val_metrics['f1']

    i = 1
    for test_list in [test_list_1, test_list_2, test_list_3, test_list_4]:
        monthly_results_path = os.path.join(result_folder, f'tif_test_file_{i}.csv')
        with open(monthly_results_path, 'w') as f:
            f.write("month,precision,recall,f1\n")
        results_f1 = []
        results_f1.append(f1_val)
        for month in test_list:
            number = str(i)
            file_path = os.path.join(data_folder, f"test_features_round_{number}" ,f"{month}.pkl")
            x_test, y_test = utils.load_single_month_data(file_path)
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
        i += 1

        



if __name__ == "__main__":

    train_path = os.path.join(data_folder, "all_train_features.pkl")

    """
    train base line models
    """
    # drebin
    # eval_svm(train_path)

    # deepdrebin
    # eval_deepdrebin(train_path)

    # t-stability
    # eval_t_stability()

    # tif data preparation
    # utils.get_train_dataset_envs(train_path, type='year')

    # tif:mpc stage 1
    # best_model_path = None
    # eval_mpc_stage_1(train_path, best_model_path=best_model_path)

    # tif:mpc stage 2
    # best_model_path = None
    # eval_mpc_stage_2(train_path, best_model_path=best_model_path)

    # tif:overall (load stg1 first - reset optimizer - train stg2)

    # best_model_path = "/root/malware/ELSA/checkpoints/model_epoch49_lr0.0005_bs1024_f1_0.966_20250222_110238.pt"
    # eval_tif(train_path, best_stg1_model_path="/root/malware/ELSA/checkpoints/model_epoch46_lr0.0005_bs1024_f1_0.961_20250227_053620.pt", best_stg2_model_path=None)



    