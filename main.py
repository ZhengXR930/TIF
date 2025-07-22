
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
from stage1_trainer import Stg1CustomDataset, St1ModelTrainer
from stage2_trainer import Stg2CustomDataset, St2ModelTrainer
from model import DrebinMLP, DrebinMLP_IRM
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import base_line.drebin as drebin
import utils
import t_stability as ts


data_folder = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features/"
result_folder = "/cs/academic/phd3/xinrzhen/xinran/SaTML/results"
save_folder = "/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt"


    
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
        
        best_model_path = trainer.train(x_train, x_val, y_train, y_val, epochs=60)
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
                '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']

    """
    train base line models
    """
    # drebin
    # eval_svm(train_path, val_path, test_list)

    # deepdrebin
    eval_deepdrebin(train_path, test_list, best_model_path=None)

    # t-stability
    # eval_t_stability()

    # mpc
    # eval_mpc(train_path, test_list, best_model_path=None)

    # tif data preparation
    # utils.get_train_dataset_envs(train_path, type='year')

    # tif:mpc stage 1
    # best_model_path = None
    # eval_mpc_stage_1(train_path, test_list, best_model_path=best_model_path)

    # tif:mpc stage 2
    # best_model_path = None
    # eval_mpc_stage_2(train_path, best_model_path=best_model_path)

    # tif:overall (load stg1 first - reset optimizer - train stg2)

    # best_model_path = "/root/malware/ELSA/checkpoints/model_epoch49_lr0.0005_bs1024_f1_0.966_20250222_110238.pt"
    # eval_tif(train_path, best_stg1_model_path="/root/malware/ELSA/checkpoints/model_epoch46_lr0.0005_bs1024_f1_0.961_20250227_053620.pt", best_stg2_model_path=None)



    