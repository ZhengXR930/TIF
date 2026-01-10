
import sys
import os
# Add the directory containing this script to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
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
import argparse


def set_seed(seed):
    """Set random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def eval_svm(train_path, val_path, test_list, data_folder, result_folder):
    x_train, y_train, _, _ = utils.load_train_overall(train_path)
    x_val, y_val, _, _ = utils.load_train_overall(val_path)
    clf = drebin.drebin_svm_train(x_train, y_train)
    _, _, f1 = drebin.drebin_svm_pred(clf, x_val, y_val)

    result_f1 = []
    # result_f1.append(f1)
    drebin.drebin_svm_monthly(clf,result_f1, result_folder, data_folder, test_list, f'svm_test_file.csv')


def eval_t_stability(test_list, data_folder, result_folder, save_folder, seed=1):
    ts.creat_t_stability()
    ts_path = os.path.join(save_folder, "t_stability.pkl")
    w, b, f1 = ts.retrain_svm(ts_path)

    i = seed
    results_f1 = []
    # results_f1.append(f1)
    with open(os.path.join(result_folder, f'ts_test_file_{i}.csv'), 'w') as f:
        f.write("month,precision,recall,f1,aut\n")
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


def eval_deepdrebin(train_path, val_path, test_list, data_folder, result_folder, save_folder, 
                    best_model_path=None, device='cuda', batch_size=128, learning_rate=0.0001, epochs=30):
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
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=save_folder)
        
        best_model_path = trainer.train(x_train, x_val, y_train, y_val, epochs=epochs)
        print(f"best model path: {best_model_path}")
    else:
        model = ModelTrainer.load_model(
        model_path=best_model_path,
        model_class=DrebinMLP,
        input_size=input_size
        )

        trainer = ModelTrainer(
        model=model,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=save_folder)

    val_dataset = CustomDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)
    f1_val = val_metrics['f1']

    monthly_results_path = os.path.join(result_folder, f'deep_test_file.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1,aut\n")
    results_f1 = []
    # results_f1.append(f1_val)
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}")



def eval_mpc_stage_1(train_path, val_path, test_list, data_folder, result_folder, save_folder,
                      best_model_path=None, device='cuda', batch_size=256, learning_rate=0.0001, 
                      con_loss_weight=1.0, epochs=30, eval_batch_size=128,
                      use_multi_proxy=True, n_proxy=3, weight_decay=1e-4, 
                      proxy_lr_multiplier=1.0, use_scheduler=False, early_stop_patience=100):
    """
    Evaluate Stage 1 model with optional multi-proxy support.
    
    Args:
        use_multi_proxy: If True, each environment learns its own proxy (paper setting)
        n_proxy: Number of proxies per class
        Other parameters: standard training parameters
    """
    x_train, y_train, env_train, t_train = utils.load_train_overall(train_path)
    x_val, y_val, env_val, t_val = utils.load_train_overall(val_path)
    input_size = x_train.shape[1]

    if best_model_path is None:
        model = DrebinMLP_IRM(input_size=input_size)
        trainer = St1ModelTrainer(
            model=model,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            con_loss_weight=con_loss_weight,
            save_dir=save_folder,
            use_multi_proxy=use_multi_proxy,
            n_proxy=n_proxy,
            weight_decay=weight_decay,
            proxy_lr_multiplier=proxy_lr_multiplier,
            use_scheduler=use_scheduler,
            early_stop_patience=early_stop_patience)

        best_model_path = trainer.train(x_train, x_val, y_train, y_val, env_train, env_val, epochs=epochs)
        print(f"best model path: {best_model_path}")
    
    else:
        print(f"load best model from {best_model_path}")
        result = St1ModelTrainer.load_model(
            model_path=best_model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size
        )
        if len(result) == 2 and result[1] is not None:
            # Multi-proxy checkpoint
            model, env_losses = result
            use_multi_proxy = True
        else:
            # Single-proxy checkpoint
            model = result[0] if isinstance(result, tuple) else result
            env_losses = None
            use_multi_proxy = False
        
        trainer = St1ModelTrainer(
            model=model,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            con_loss_weight=con_loss_weight,
            save_dir=save_folder,
            use_multi_proxy=use_multi_proxy)
    
    val_dataset = Stg1CustomDataset(x_val, y_val, env_val)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)
    f1_val = val_metrics['f1']

    monthly_results_path = os.path.join(result_folder, f'stage1_test_file.csv')
    if use_multi_proxy:
        monthly_results_path = os.path.join(result_folder, f'stage1_multi_proxy_test_file.csv')
    
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1,aut\n")
    results_f1 = []
    # results_f1.append(f1_val)
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}, AUT: {m_aut:.4f}")


def eval_mpc_stage_2(train_path, val_path, test_list, data_folder, result_folder, save_folder,
                     best_model_path=None, device='cuda', batch_size=512, learning_rate=0.0001, 
                     con_loss_weight=1.0, penalty_weight=0.05, epochs=100, eval_batch_size=128, seed=1):
    x_train, y_train, env_train, t_train = utils.load_train_overall(train_path)
    x_val, y_val, env_val, t_val = utils.load_train_overall(val_path)
    input_size = x_train.shape[1]

    if best_model_path is None:
        model = DrebinMLP_IRM(input_size=input_size)
        trainer = St2ModelTrainer(
            model=model,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            con_loss_weight=con_loss_weight,
            penalty_weight=penalty_weight,
            save_dir=save_folder)

        best_model_path = trainer.train(x_train, x_val, y_train, y_val, env_train, env_val, epochs=epochs)
        print(f"best model path: {best_model_path}")
    
    else:
        print(f"load best model from {best_model_path}")
        result = St2ModelTrainer.load_model(
            model_path=best_model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size
        )
        if len(result) == 3:
            model, custom_loss_state_dict, n_proxy = result
        else:
            model, custom_loss_state_dict = result
            n_proxy = None
        trainer = St2ModelTrainer(
            model=model,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            con_loss_weight=con_loss_weight,
            penalty_weight=penalty_weight,
            save_dir=save_folder,
            custom_loss_state_dict=custom_loss_state_dict,
            n_proxy=n_proxy if n_proxy is not None else 5
        )
    
    val_dataset = Stg2CustomDataset(x_val, y_val, env_val)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)


    monthly_results_path = os.path.join(result_folder, f'stage2_test_file_{seed}.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1,aut\n")
    results_f1 = []
    # results_f1.append(f1_val)
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}")


def eval_tif(train_path, val_path, test_list, data_folder, result_folder, save_folder,
             best_stg1_model_path=None, best_stg2_model_path=None, device='cuda', batch_size=1024, 
             learning_rate=0.0001, con_loss_weight=0.1, penalty_weight=1.0, epochs=20, 
             eval_batch_size=128, mpc_load_mode='full', weight_decay=1e-3, 
             stage1_n_proxy=3, stage2_n_proxy=3,
             early_stop_patience=5):
    x_train, y_train, env_train, t_train = utils.load_train_overall(train_path)
    x_val, y_val, env_val, t_val = utils.load_train_overall(val_path)

    input_size = x_train.shape[1]

    # If best_stg2_model_path is provided, we can skip stage1 and directly load stage2 for inference
    if best_stg2_model_path is not None:
        print(f"load best stg2 model from {best_stg2_model_path}")
        result = St2ModelTrainer.load_model(
            model_path=best_stg2_model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size,
            device=device
        )
        if len(result) == 3:
            model, custom_loss_state_dict, n_proxy = result
        else:
            model, custom_loss_state_dict = result
            n_proxy = None
        trainer = St2ModelTrainer(
            model=model,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            con_loss_weight=con_loss_weight,
            penalty_weight=penalty_weight,
            save_dir=save_folder,
            custom_loss_state_dict=custom_loss_state_dict
        )
    
    elif best_stg1_model_path is None:
        print("Error: Either best_stg1_model_path or best_stg2_model_path must be provided!")
        return
    
    elif best_stg2_model_path is None:
        print(f"load best stg1 model from {best_stg1_model_path}")
        # Load both model weights and MPC proxy parameters from stage 1 checkpoint
        result = St2ModelTrainer.load_model(
            model_path=best_stg1_model_path,
            model_class=DrebinMLP_IRM,
            input_size=input_size
        )
        if len(result) == 3:
            model, custom_loss_state_dict, stg1_n_proxy = result
        else:
            model, custom_loss_state_dict = result
            stg1_n_proxy = None
        
        # MPC loading strategy:
        # - 'full': Load all (proj + proxies) - use when TIF data distribution is similar to stage1
        # - 'proj_only': Load only projection layer, reinit proxies - use when distributions differ
        # - 'none': Don't load, reinit all - use when starting fresh or distributions very different
        # Note: If stage1 and stage2 have different n_proxy, use 'proj_only' or 'none'
        if stg1_n_proxy is not None and stg1_n_proxy != stage2_n_proxy:
            if mpc_load_mode == 'full':
                print(f"Warning: Stage1 n_proxy={stg1_n_proxy} != Stage2 n_proxy={stage2_n_proxy}")
                print("Switching to 'proj_only' mode to avoid proxy count mismatch")
                mpc_load_mode = 'proj_only'
        
        trainer = St2ModelTrainer(
            model=model,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,  
            con_loss_weight=con_loss_weight,
            penalty_weight=penalty_weight,
            save_dir=save_folder,
            custom_loss_state_dict=custom_loss_state_dict,
            mpc_load_mode=mpc_load_mode,
            weight_decay=weight_decay,
            n_proxy=stage2_n_proxy
        )

        trainer.reset_optimizer(learning_rate=learning_rate)
        best_model_path = trainer.train(x_train, x_val, y_train, y_val, env_train, env_val, epochs=epochs)
        print(f"best model path: {best_model_path}")

    val_dataset = Stg2CustomDataset(x_val, y_val, env_val)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)


    monthly_results_path = os.path.join(result_folder, f'tif_test_file_{seed}.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1,aut\n")
    results_f1 = []
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}")


def eval_mpc(train_path, val_path, test_list, data_folder, result_folder, save_folder,
             best_model_path=None, device='cuda', batch_size=256, learning_rate=0.0001, 
             con_loss_weight=1.0, epochs=60, eval_batch_size=128):
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
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        con_loss_weight=con_loss_weight,
        save_dir=save_folder)
        
        best_model_path = trainer.train(x_train, x_val, y_train, y_val, epochs=epochs)
        print(f"best model path: {best_model_path}")
    else:
        model = ModelTrainer.load_model(
        model_path=best_model_path,
        model_class=DrebinMLP_IRM,
        input_size=input_size
        )

        trainer = ModelTrainer(
        model=model,
        device=device,
        batch_size=eval_batch_size,
        learning_rate=learning_rate,
        save_dir=save_folder)

    val_dataset = CustomDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_metrics = trainer.evaluate(val_loader)

    monthly_results_path = os.path.join(result_folder, f'mpc_test_file.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1,aut\n")
    results_f1 = []
    # results_f1.append(f1_val)
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        test_metrics = trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}")   



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SaTML: Training and Evaluation Script')
    
    # Method and mode selection
    parser.add_argument('--method', type=str, required=True,
                        choices=['tif', 'drebin'],
                        help='Method to use: tif or drebin')
    parser.add_argument('--mode', type=str, required=True,
                        help='Mode to run. For tif: stage1, stage2, tif, mpc. For drebin: svm, deep, ts')
    
    # Paths
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to data folder')
    parser.add_argument('--result_folder', type=str, required=True,
                        help='Path to result folder')
    parser.add_argument('--save_folder', type=str, required=True,
                        help='Path to save checkpoints')
    
    # Model paths
    parser.add_argument('--best_model_path', type=str, default=None,
                        help='Path to best model checkpoint (optional)')
    parser.add_argument('--best_stg1_model_path', type=str, default=None,
                        help='Path to best stage 1 model checkpoint (for tif mode)')
    parser.add_argument('--best_stg2_model_path', type=str, default=None,
                        help='Path to best stage 2 model checkpoint (for tif mode)')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    
    # TIF-specific parameters
    parser.add_argument('--con_loss_weight', type=float, default=1.0,
                        help='Contrastive loss weight')
    parser.add_argument('--penalty_weight', type=float, default=1.0,
                        help='Penalty weight for stage 2 (default: 1.0)')
    parser.add_argument('--mpc_load_mode', type=str, default='full',
                        choices=['full', 'proj_only', 'none', 'auto'],
                        help='MPC loading mode for TIF')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for optimizer (default: 0 for stage1, 1e-3 for stage2/tif)')
    parser.add_argument('--use_multi_proxy', action='store_true', default=True,
                        help='Use multi-proxy mode for Stage 1 (default: True, each environment learns its own proxy)')
    parser.add_argument('--stage1_n_proxy', type=int, default=3,
                        help='Number of proxies per class in Stage 1 (default: 3)')
    parser.add_argument('--stage2_n_proxy', type=int, default=3,
                        help='Number of proxies per class in Stage 2 (default: 3)')
    parser.add_argument('--early_stop_patience', type=int, default=100,
                        help='Early stopping patience (default: 100 for stage1, 5 for stage2/tif)')
    
    # Test list (can be provided as comma-separated string or use default)
    parser.add_argument('--test_list', type=str, default=None,
                        help='Comma-separated list of test months (e.g., "2015-01,2015-02") or path to file with one month per line')
    
    args = parser.parse_args()
    
    # Validate mode based on method
    if args.method == 'tif':
        if args.mode not in ['stage1', 'stage2', 'tif', 'mpc']:
            parser.error(f"Mode '{args.mode}' is not valid for method 'tif'. Must be one of: stage1, stage2, tif, mpc")
    elif args.method == 'drebin':
        if args.mode not in ['svm', 'deep', 'ts']:
            parser.error(f"Mode '{args.mode}' is not valid for method 'drebin'. Must be one of: svm, deep, ts")
    
    # Parse test list
    if args.test_list is None:
        # Default test list
        args.test_list = ['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
                         '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12',
                         '2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
                         '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12',
                         '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
                         '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12',
                         '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
                         '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
                         '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12', 
                         '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12',
                         '2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06']
    else:
        # Check if it's a file path
        if os.path.isfile(args.test_list):
            with open(args.test_list, 'r') as f:
                args.test_list = [line.strip() for line in f if line.strip()]
        else:
            # Comma-separated string
            args.test_list = [m.strip() for m in args.test_list.split(',')]
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup paths
    data_folder = args.data_folder
    result_folder = args.result_folder
    save_folder = args.save_folder
    
    # Create directories if they don't exist
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    
    train_path = os.path.join(data_folder, "train_data.pkl")
    val_path = os.path.join(data_folder, "val_data.pkl")
    test_list = args.test_list
    
    # Run based on method and mode
    if args.method == 'tif':
        if args.mode == 'stage1':
            # Stage 1 defaults: use_multi_proxy=True, weight_decay=1e-4, con_loss_weight=1.0, epochs=30, n_proxy=3, early_stop_patience=100
            eval_mpc_stage_1(
                train_path=train_path,
                val_path=val_path,
                test_list=test_list,
                data_folder=data_folder,
                result_folder=result_folder,
                save_folder=save_folder,
                best_model_path=args.best_model_path,
                device=args.device,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                con_loss_weight=args.con_loss_weight,
                epochs=30 if args.epochs == 100 else args.epochs,  # Default 30 for stage1
                eval_batch_size=args.eval_batch_size,
                use_multi_proxy=args.use_multi_proxy,
                n_proxy=args.stage1_n_proxy,
                weight_decay=1e-4 if args.weight_decay == 0 else args.weight_decay,  # Default 1e-4 for stage1
                proxy_lr_multiplier=1.0,
                use_scheduler=False,
                early_stop_patience=100 if args.early_stop_patience == 5 else args.early_stop_patience  # Default 100 for stage1
            )
        elif args.mode == 'stage2':
            eval_mpc_stage_2(
                train_path=train_path,
                val_path=val_path,
                test_list=test_list,
                data_folder=data_folder,
                result_folder=result_folder,
                save_folder=save_folder,
                best_model_path=args.best_model_path,
                device=args.device,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                con_loss_weight=args.con_loss_weight,
                penalty_weight=args.penalty_weight,
                epochs=args.epochs,
                eval_batch_size=args.eval_batch_size,
                seed=args.seed
            )
        elif args.mode == 'tif':
            # For TIF mode, either best_stg1_model_path or best_stg2_model_path must be provided
            # If best_stg2_model_path is provided, we can skip stage1 and directly load stage2 for inference
            if args.best_stg1_model_path is None and args.best_stg2_model_path is None:
                raise ValueError("Either --best_stg1_model_path or --best_stg2_model_path is required for tif mode")
            # Stage 2/TIF defaults: con_loss_weight=0.1, penalty_weight=1.0, epochs=20, weight_decay=1e-3, batch_size=1024, n_proxy=3
            eval_tif(
                train_path=train_path,
                val_path=val_path,
                test_list=test_list,
                data_folder=data_folder,
                result_folder=result_folder,
                save_folder=save_folder,
                best_stg1_model_path=args.best_stg1_model_path,
                best_stg2_model_path=args.best_stg2_model_path,
                device=args.device,
                batch_size=args.batch_size,  # Default 1024 for stage2
                learning_rate=args.learning_rate,
                con_loss_weight=args.con_loss_weight,  # Default 0.1 for stage2
                penalty_weight=args.penalty_weight,  # Default 1.0
                epochs=args.epochs,  # Default 20 for stage2
                eval_batch_size=args.eval_batch_size,
                mpc_load_mode=args.mpc_load_mode,
                weight_decay=args.weight_decay,  # Default 1e-3 for stage2
                stage1_n_proxy=args.stage1_n_proxy,  # Default 3
                stage2_n_proxy=args.stage2_n_proxy,  # Default 3
                early_stop_patience=args.early_stop_patience  # Default 5 for stage2
            )
        elif args.mode == 'mpc':
            eval_mpc(
                train_path=train_path,
                val_path=val_path,
                test_list=test_list,
                data_folder=data_folder,
                result_folder=result_folder,
                save_folder=save_folder,
                best_model_path=args.best_model_path,
                device=args.device,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                con_loss_weight=args.con_loss_weight,
                epochs=args.epochs,
                eval_batch_size=args.eval_batch_size
            )
    
    elif args.method == 'drebin':
        if args.mode == 'svm':
            eval_svm(
                train_path=train_path,
                val_path=val_path,
                test_list=test_list,
                data_folder=data_folder,
                result_folder=result_folder
            )
        elif args.mode == 'deep':
            eval_deepdrebin(
                train_path=train_path,
                val_path=val_path,
                test_list=test_list,
                data_folder=data_folder,
                result_folder=result_folder,
                save_folder=save_folder,
                best_model_path=args.best_model_path,
                device=args.device,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs
            )
        elif args.mode == 'ts':
            eval_t_stability(
                test_list=test_list,
                data_folder=data_folder,
                result_folder=result_folder,
                save_folder=save_folder,
                seed=args.seed
            )