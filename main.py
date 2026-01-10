
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

    return seed
    
def eval_svm(train_path, val_path, test_list, data_folder, result_folder):
    x_train, y_train, _, _ = utils.load_train_overall(train_path)
    x_val, y_val, _, _ = utils.load_train_overall(val_path)
    clf = drebin.drebin_svm_train(x_train, y_train)
    _, _, f1 = drebin.drebin_svm_pred(clf, x_val, y_val)

    result_f1 = []
    drebin.drebin_svm_monthly(clf,result_f1, result_folder, data_folder, test_list, f'svm_test_file.csv')


def eval_t_stability(test_list, data_folder, result_folder, save_folder, seed=1):
    ts.creat_t_stability()
    ts_path = os.path.join(save_folder, "t_stability.pkl")
    w, b, f1 = ts.retrain_svm(ts_path)

    results_f1 = []
    with open(os.path.join(result_folder, f'ts_test_file_{seed}.csv'), 'w') as f:
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
        with open(os.path.join(result_folder, f'ts_test_file_{seed}.csv'), 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")


def eval_deepdrebin(train_path, val_path, test_list, data_folder, result_folder, save_folder, 
                    best_model_path=None, device='cuda', batch_size=128, learning_rate=0.0001, epochs=30, seed=1):
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

    monthly_results_path = os.path.join(result_folder, f'deep_test_file_{seed}.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1,aut\n")
    results_f1 = []

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



def eval_tif(train_path, val_path, test_list, data_folder, result_folder, save_folder,
                        stage1_batch_size=512, stage2_batch_size=1024,
                        stage1_learning_rate=0.0001, stage2_learning_rate=0.0001,
                        stage1_con_loss_weight=1.0, stage2_con_loss_weight=0.1,
                        stage1_weight_decay=0, stage2_weight_decay=1e-3,
                        stage1_epochs=30, stage2_epochs=20,
                        stage1_n_proxy=3, stage2_n_proxy=3,
                        stage1_early_stop_patience=100, stage2_early_stop_patience=5,
                        penalty_weight=1.0, mpc_load_mode='full', device='cuda', eval_batch_size=128, seed=1):

    x_train, y_train, env_train, t_train = utils.load_train_overall(train_path)
    x_val, y_val, env_val, t_val = utils.load_train_overall(val_path)
    input_size = x_train.shape[1]
    
    print("=" * 80)
    print("TIF Training")
    print("=" * 80)
    print(f"Stage 1: Multi-proxy training ({stage1_epochs} epochs)")
    print(f"  - batch_size: {stage1_batch_size}, lr: {stage1_learning_rate}")
    print(f"  - con_loss_weight: {stage1_con_loss_weight}, weight_decay: {stage1_weight_decay}")
    print(f"Stage 2: Fused proxy + IRM training ({stage2_epochs} epochs)")
    print(f"  - batch_size: {stage2_batch_size}, lr: {stage2_learning_rate}")
    print(f"  - con_loss_weight: {stage2_con_loss_weight}, penalty_weight: {penalty_weight}")
    print(f"  - weight_decay: {stage2_weight_decay}")
    print("=" * 80)
    
    # ========== Stage 1: Multi-Proxy Training ==========
    print("\n" + "=" * 80)
    print("Stage 1: Discriminative Information Amplification")
    print("=" * 80)
    
    model = DrebinMLP_IRM(input_size=input_size)
    stage1_trainer = St1ModelTrainer(
        model=model,
        device=device,
        batch_size=stage1_batch_size,
        learning_rate=stage1_learning_rate,
        con_loss_weight=stage1_con_loss_weight,
        save_dir=save_folder,
        use_multi_proxy=True,
        n_proxy=stage1_n_proxy,
        weight_decay=stage1_weight_decay,
        proxy_lr_multiplier=1.0,
        use_scheduler=False,
        early_stop_patience=stage1_early_stop_patience
    )
    
    # Train Stage 1 (use final model state, not best model)
    stage1_trainer.train(x_train, x_val, y_train, y_val, env_train, env_val, epochs=stage1_epochs)
    print(f"\nStage 1 completed. Using final model state (last epoch) for Stage 2.")
    
    # ========== Extract and fuse multi-proxy to single proxy ==========
    print("\n" + "=" * 80)
    print("Stage 2: Unstable Information Suppression")
    print("=" * 80)
    print("Fusing multi-proxy from Stage 1 to single proxy for Stage 2...")
    
    if stage1_trainer.custom_losses is None or len(stage1_trainer.custom_losses) == 0:
        raise ValueError("Stage 1 must use multi-proxy mode (use_multi_proxy=True)")
    
    env_losses_state_dict = {}
    for env_id, env_loss in stage1_trainer.custom_losses.items():
        env_losses_state_dict[env_id] = env_loss.state_dict()
    
    custom_loss_state_dict = St2ModelTrainer._fuse_multi_proxy_stage1(env_losses_state_dict)
    
    if stage1_n_proxy != stage2_n_proxy:
        if mpc_load_mode == 'full':
            print(f"Warning: Stage1 n_proxy={stage1_n_proxy} != Stage2 n_proxy={stage2_n_proxy}")
            print("Switching to 'proj_only' mode to avoid proxy count mismatch")
            mpc_load_mode = 'proj_only'
    
    # ========== Stage 2: Fused Proxy + IRM Training ==========
    stage2_trainer = St2ModelTrainer(
        model=stage1_trainer.model, 
        device=device,
        batch_size=stage2_batch_size,  
        learning_rate=stage2_learning_rate,  
        con_loss_weight=stage2_con_loss_weight,  
        penalty_weight=penalty_weight,
        save_dir=save_folder,
        custom_loss_state_dict=custom_loss_state_dict,
        mpc_load_mode=mpc_load_mode,
        weight_decay=stage2_weight_decay,  
        n_proxy=stage2_n_proxy,
        early_stop_patience=stage2_early_stop_patience
    )
    
    stage2_trainer.reset_optimizer(learning_rate=stage2_learning_rate)
    best_stg2_model_path = stage2_trainer.train(
        x_train, x_val, y_train, y_val, env_train, env_val, epochs=stage2_epochs
    )
    print(f"\nStage 2 completed. Best model saved: {best_stg2_model_path}")
    
    # ========== Evaluation ==========
    print("\n" + "=" * 80)
    print("Evaluating on test sets...")
    print("=" * 80)

    
    monthly_results_path = os.path.join(result_folder, f'tif_test_file_{seed}.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1,aut\n")
    results_f1 = []
    
    for month in test_list:
        file_path = os.path.join(data_folder, f"{month}.pkl")
        x_test, y_test, env_test = utils.load_single_month_data(file_path)
        from scipy import sparse
        if sparse.issparse(x_test):
            x_test = x_test.toarray()
        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        test_metrics = stage2_trainer.evaluate(test_loader)
        precision = test_metrics['precision']
        recall = test_metrics['recall']
        f1 = test_metrics['f1']
        results_f1.append(f1)
        m_aut = tm.aut(results_f1)
        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")
        print(f"test month: {month}, test metrics: {test_metrics}, AUT: {m_aut:.4f}")
    
    return best_stg2_model_path




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SaTML: Training and Evaluation Script')
    
    # Method and mode selection
    parser.add_argument('--method', type=str, required=True,
                        choices=['tif', 'drebin'],
                        help='Method to use: tif or drebin')
    parser.add_argument('--mode', type=str, required=True,
                        help='Mode to run. For tif: tif. For drebin: svm, deep, ts')
    
    # Paths
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to data folder')
    parser.add_argument('--result_folder', type=str, required=True,
                        help='Path to result folder')
    parser.add_argument('--save_folder', type=str, required=True,
                        help='Path to save checkpoints')
    
    # Model paths
    parser.add_argument('--best_model_path', type=str, default=None,
                        help='Path to best model checkpoint (optional, for drebin methods)')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    
    # TIF specific parameters
    parser.add_argument('--stage1_batch_size', type=int, default=512,
                        help='Batch size for Stage 1 training (default: 512)')
    parser.add_argument('--stage2_batch_size', type=int, default=1024,
                        help='Batch size for Stage 2 training (default: 1024)')
    parser.add_argument('--stage1_learning_rate', type=float, default=0.0001,
                        help='Learning rate for Stage 1 (default: 0.0001)')
    parser.add_argument('--stage2_learning_rate', type=float, default=0.0001,
                        help='Learning rate for Stage 2 (default: 0.0001)')
    parser.add_argument('--stage1_con_loss_weight', type=float, default=1.0,
                        help='Contrastive loss weight for Stage 1 (default: 1.0)')
    parser.add_argument('--stage2_con_loss_weight', type=float, default=0.1,
                        help='Contrastive loss weight for Stage 2 (default: 0.1)')
    parser.add_argument('--stage1_weight_decay', type=float, default=0,
                        help='Weight decay for Stage 1 (default: 0)')
    parser.add_argument('--stage2_weight_decay', type=float, default=1e-3,
                        help='Weight decay for Stage 2 (default: 1e-3)')
    parser.add_argument('--stage1_epochs', type=int, default=30,
                        help='Number of epochs for Stage 1 (default: 30)')
    parser.add_argument('--stage2_epochs', type=int, default=20,
                        help='Number of epochs for Stage 2 (default: 20)')
    parser.add_argument('--stage1_n_proxy', type=int, default=3,
                        help='Number of proxies per class in Stage 1 (default: 3)')
    parser.add_argument('--stage2_n_proxy', type=int, default=3,
                        help='Number of proxies per class in Stage 2 (default: 3)')
    parser.add_argument('--stage1_early_stop_patience', type=int, default=100,
                        help='Early stopping patience for Stage 1 (default: 100)')
    parser.add_argument('--stage2_early_stop_patience', type=int, default=5,
                        help='Early stopping patience for Stage 2 (default: 5)')
    parser.add_argument('--penalty_weight', type=float, default=1.0,
                        help='IRM penalty weight for Stage 2 (default: 1.0)')
    parser.add_argument('--mpc_load_mode', type=str, default='full',
                        choices=['full', 'proj_only', 'none', 'auto'],
                        help='MPC loading mode for Stage 2 (default: full)')
    
    # Drebin-specific parameters (for backward compatibility)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training (for drebin methods)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate (for drebin methods)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (for drebin methods)')
    
    # Test list (can be provided as comma-separated string or use default)
    parser.add_argument('--test_list', type=str, default=None,
                        help='Comma-separated list of test months (e.g., "2015-01,2015-02") or path to file with one month per line')
    
    args = parser.parse_args()
    
    # Validate mode based on method
    if args.method == 'drebin':
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
        if os.path.isfile(args.test_list):
            with open(args.test_list, 'r') as f:
                args.test_list = [line.strip() for line in f if line.strip()]
        else:
            args.test_list = [m.strip() for m in args.test_list.split(',')]
    
    return args


if __name__ == "__main__":
    args = parse_args()
    

    set_seed(args.seed)
    
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
        if args.mode == 'tif':
            # TIF training: Stage 1 -> Stage 2
            eval_tif(
                train_path=train_path,
                val_path=val_path,
                test_list=test_list,
                data_folder=data_folder,
                result_folder=result_folder,
                save_folder=save_folder,
                stage1_batch_size=args.stage1_batch_size,
                stage2_batch_size=args.stage2_batch_size,
                stage1_learning_rate=args.stage1_learning_rate,
                stage2_learning_rate=args.stage2_learning_rate,
                stage1_con_loss_weight=args.stage1_con_loss_weight,
                stage2_con_loss_weight=args.stage2_con_loss_weight,
                stage1_weight_decay=args.stage1_weight_decay,
                stage2_weight_decay=args.stage2_weight_decay,
                stage1_epochs=args.stage1_epochs,
                stage2_epochs=args.stage2_epochs,
                stage1_n_proxy=args.stage1_n_proxy,
                stage2_n_proxy=args.stage2_n_proxy,
                stage1_early_stop_patience=args.stage1_early_stop_patience,
                stage2_early_stop_patience=args.stage2_early_stop_patience,
                penalty_weight=args.penalty_weight,
                mpc_load_mode=args.mpc_load_mode,
                device=args.device,
                eval_batch_size=args.eval_batch_size,
                seed=args.seed
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
                epochs=args.epochs,
                seed=args.seed
            )
        elif args.mode == 'ts':
            eval_t_stability(
                test_list=test_list,
                data_folder=data_folder,
                result_folder=result_folder,
                save_folder=save_folder,
                seed=args.seed
            )