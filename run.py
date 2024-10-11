########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/microsoft/Drug-Interaction-Research/tree/DSN-DDI-for-DDI-Prediction
# https://github.com/thinng/GraphDTA

########################################################################################################################
########## Import
########################################################################################################################

import os
import time
import copy
import utils
import torch
import models
import logging
import warnings
import argparse
import pandas as pd
from tqdm import tqdm
from metrics import *
from rdkit import Chem
from torch import optim
from pathlib import Path
from datetime import date
import torch.nn.functional as F
from preprocessing import CustomDataset, CustomDataLoader

########################################################################################################################
########## Pre-settings
########################################################################################################################

tqdm.pandas()
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

loss_dict = {'DTA': F.mse_loss, 'PPI': F.binary_cross_entropy}

metrics_dict = {'DTA': {'MSE': mse, 'RMSE': rmse, 'CI': ci, 'RM2': rm2,
                        'Pearson': pearson, 'Spearman': spearman},
                'PPI': {'ACC': accuracy, 'AUC': auc_score, 'Precision': precision,
                        'Recall': recall, 'F1-score': f1_score, 'AUPR': aupr}}

########################################################################################################################
########## Functions
########################################################################################################################


def valid_drug(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol


def valid_protein(protein, dn, prot_inform):
    key, seq = protein
    fd = Path(f'TDC/DTA/{dn}/protein_graph_pyg')
    find_idx = prot_inform.index[prot_inform.eq(key).any(axis=1)]
    if not find_idx.empty:
        find_row = prot_inform.loc[find_idx, :].to_dict('records')[0]
        if seq == find_row['Seq']:
            for k, v in find_row.items():
                if k != 'Seq':
                    file = fd / f"{v}.pt"
                    if file.is_file():
                        return torch.load(file)


def cal_perform(real, pred, dt_name, lfs):
    result = {'Set': dt_name}
    real = real.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    for lfn, lf in lfs.items():
        result[lfn] = float(f"{lf(real, pred):.3f}")
    return result



# training function at each epoch
def train(model, device, train_loader, loss_fn, optimizer, epoch, task='reg'):
    start = time.time()
    model.train()

    train_loss = 0
    processed_data = 0
    train_preds = torch.Tensor()
    train_reals = torch.Tensor()
    for batch_idx, data in enumerate(train_loader):

        data = [d.to(device) for d in data]

        processed_data += len(data[0])
        optimizer.zero_grad()

        pred, real = model(data)
        loss = loss_fn(pred, real)

        task_pred = (F.sigmoid(pred) >= 0.5).int() if task == 'cls' else pred
        train_preds = torch.cat((train_preds, task_pred.cpu()), 0)
        train_reals = torch.cat((train_reals, real.cpu()), 0)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        trn_dt_len = len(trn_loader.dataset)
        processed_percent = 100. * processed_data / trn_dt_len
        # if (processed_percent > 50 and log_signal) or (processed_percent == 100):
        if processed_percent == 100:
            runtime = f"{(time.time() - start) / 60:.2f} min"
            logger.info('Train epoch ({}): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(runtime, epoch,
                                                                                 processed_data,
                                                                                 trn_dt_len,
                                                                                 processed_percent,
                                                                                 loss.item()))
    train_loss = train_loss / len(train_loader)
    return train_preds, train_reals, train_loss


def evaluation(model, device, loader, task='reg'):
    model.eval()
    start = time.time()
    total_preds = torch.Tensor()
    total_reals = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = [d.to(device) for d in data]
            pred, real = model(data)
            task_pred = (F.sigmoid(pred) >= 0.5).int() if task == 'cls' else pred

            total_preds = torch.cat((total_preds, task_pred.cpu()), 0)
            total_reals = torch.cat((total_reals, real.cpu()), 0)
    runtime = f"{(time.time() - start) / 60:.2f} min"
    logger.info(f'eval runtime ({runtime})')
    return total_preds, total_reals


########################################################################################################################
########## Run script
########################################################################################################################

if __name__ == '__main__':

    ####################################################################################################################
    ########## Parameters
    ####################################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='dataset')
    parser.add_argument('--d1_col', type=str, required=True, help='data1 column name')
    parser.add_argument('--d2_col', type=str, required=True, help='data2 column name')
    parser.add_argument('--d1_type', type=str, required=True, help='data1 type')
    parser.add_argument('--d2_type', type=str, required=True, help='data2 type')
    parser.add_argument('--data_name', type=str, required=True, help='dataset name')
    parser.add_argument('--task_name', type=str, required=True, help='task (DTA or PPI)')
    parser.add_argument('--project_name', type=str, required=True, help='project name')

    parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
    parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
    parser.add_argument('--n_workers', type=int, default=1, help='num of workers for dataset')
    parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')

    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])

    args = parser.parse_args()

    data_folder = Path(args.data)
    d1_col = args.d1_col
    d2_col = args.d2_col
    d1_type = args.d1_type
    d2_type = args.d2_type
    data_name = args.data_name
    task_name = args.task_name
    project_name = args.project_name

    n_atom_feats = args.n_atom_feats
    n_atom_hid = args.n_atom_hid
    lr = args.lr
    n_epochs = args.n_epochs
    n_workers = args.n_workers
    kge_dim = args.kge_dim
    batch_size = args.batch_size

    weight_decay = args.weight_decay
    device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'

    ####################################################################################################################
    ########## Run
    ####################################################################################################################

    '''
    python run.py --data ./TDC/DTA/DAVIS/random --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_random --n_workers 4 --n_epochs 100 --lr 1e-4
    python run.py --data ./TDC/DTA/DAVIS/cold_split/Drug --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_cold_drug --n_workers 4 --n_epochs 100 --lr 1e-4
    python run.py --data ./TDC/DTA/DAVIS/cold_split/Target --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_cold_target --n_workers 4 --n_epochs 100 --lr 1e-4
    '''

    # # # tmp
    # data_folder = Path('./TDC/DTA/DAVIS/random')
    # d1_col, d2_col = 'Drug', 'Target_ID'
    # d1_type, d2_type = 'Drug', 'Protein'
    # data_name = 'davis'
    # task_name = 'DTA'
    # project_name = 'Test1'
    # batch_size = 12
    # lr = 1e-3
    # weight_decay = 5e-4
    # device = 'cpu'
    # n_atom_feats = 55
    # n_atom_hid = 128
    # n_epochs = 2
    # kge_dim = 128
    # n_workers = 1

    mode_dict = {'d1': d1_type, 'd2': d2_type, 'task': task_name}

    # output path
    today = str(date.today()).replace('-', '')
    output_folder = Path(f'Results_{project_name}_{today}')
    model_folder = output_folder / 'models'
    model_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"output_folder: {output_folder}")

    # log path
    log_fd = Path(output_folder / 'logs')
    log_fd.mkdir(parents=True, exist_ok=True)
    utils.set_log(log_fd, f'models.log')
    logger.info('Dual-View-Expansion experiments...')
    logger.info(f'data_folder: {data_folder}')
    logger.info(f'd1_col: {d1_col}, d2_col: {d2_col}')
    logger.info(f'd1_type: {d1_type}, d2_type: {d2_type}')
    logger.info(f'data_name: {data_name}')
    logger.info(f'task_name: {task_name}')
    logger.info(f'project_name: {project_name}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'lr: {lr}')
    logger.info(f'device: {device}')
    logger.info(f'weight_decay: {weight_decay}')
    logger.info(f'n_atom_feats: {n_atom_feats}')
    logger.info(f'n_atom_hid: {n_atom_hid}')
    logger.info(f'n_workers: {n_workers}')
    logger.info(f'n_epochs: {n_epochs}')
    logger.info(f'kge_dim: {kge_dim}')

    # load dataset
    df_trn = pd.read_csv(data_folder / 'train.csv')
    df_val = pd.read_csv(data_folder / 'valid.csv')
    df_tst = pd.read_csv(data_folder / 'test.csv')

    df_trn['Set'] = 'trn'
    df_val['Set'] = 'val'
    df_tst['Set'] = 'tst'

    df_total = pd.concat([df_trn, df_val, df_tst]).reset_index(drop=True)
    raw_ln = len(df_total)

    prots = pd.read_csv(Path(f'TDC/DTA/{data_name}/protein_graph_pyg/{data_name}_prot.csv'))
    df_total['d1'] = df_total[d1_col].progress_apply(valid_drug)
    df_total['d2'] = df_total[[d2_col, 'Target']].progress_apply(lambda x: valid_protein(x, data_name, prots), axis=1)
    df_total = df_total.dropna()
    df_total.to_csv(output_folder / f'{project_name}_preprocessed_data.csv', index=False, header=True)
    logger.info(f'Prepare the data: raw {raw_ln} -> preprocessed {len(df_total)} (removed {len(df_total) - raw_ln})')

    df_trn = df_total[df_total['Set'] == 'trn']
    df_val = df_total[df_total['Set'] == 'val']
    df_tst = df_total[df_total['Set'] == 'tst']

    trn_tup = [(h, t, l) for h, t, l in zip(df_trn['d1'], df_trn['d2'], df_trn['Y'])]
    val_tup = [(h, t, l) for h, t, l in zip(df_val['d1'], df_val['d2'], df_val['Y'])]
    tst_tup = [(h, t, l) for h, t, l in zip(df_tst['d1'], df_tst['d2'], df_tst['Y'])]

    # start
    start = time.time()
    total_results = []
    seeds = [5, 42, 76]
    for seed in seeds:
        logger.info(f"#####" * 20)
        utils.set_random_seeds(seed)

        # Define DataLoader
        trn_dataset = CustomDataset(trn_tup, mode=mode_dict, shuffle=True)
        val_dataset = CustomDataset(val_tup, mode=mode_dict)
        tst_dataset = CustomDataset(tst_tup, mode=mode_dict)
        # trn_dataset = CustomDataset(trn_tup[:12], mode=mode_dict, shuffle=True)
        # val_dataset = CustomDataset(val_tup[:12], mode=mode_dict)
        # tst_dataset = CustomDataset(tst_tup[:12], mode=mode_dict)
        logger.info(f"TRN: {len(trn_dataset)}, VAL: {len(val_dataset)}, TST: {len(tst_dataset)}")

        trn_loader = CustomDataLoader(trn_dataset, batch_size=batch_size, shuffle=True,
                                      worker_init_fn=utils.seed_worker, num_workers=n_workers)
        val_loader = CustomDataLoader(val_dataset, batch_size=(batch_size * 3),
                                      worker_init_fn=utils.seed_worker, num_workers=n_workers)
        tst_loader = CustomDataLoader(tst_dataset, batch_size=(batch_size * 3),
                                      worker_init_fn=utils.seed_worker, num_workers=n_workers)

        # Define model
        model = models.MVN_DDI(n_atom_feats, n_atom_hid, kge_dim, heads_out_feat_params=[64, 64, 64, 64],
                               blocks_params=[2, 2, 2, 2], task=task_name)
        model.to(device)

        loss_fn = loss_dict[task_name]
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

        # for batch in trn_loader:
        #     break
        # tmp = batch
        # data = batch
        # data = [d.to(device) for d in data]
        # pred, real = model(data)
        # loss = loss_fn(pred, real)
        # loss.backward()

        # train & evaluation
        best_epoch = -1
        performs = pd.DataFrame()
        model_state = model.state_dict()
        task_metric = metrics_dict[task_name]

        if task_name == 'DTA':
            best_loss = np.inf
            model_results = {}
            for epoch in range(n_epochs):
                trn_preds, trn_reals, trn_loss = train(model, device, trn_loader, loss_fn, optimizer, epoch + 1)
                val_preds, val_reals = evaluation(model, device, val_loader)
                val_loss = loss_fn(val_preds, val_reals).item()

                if val_loss < best_loss:
                    # save loss
                    best_loss = val_loss
                    best_epoch = epoch + 1
                    model_state = copy.deepcopy(model.state_dict())

                    # tst
                    tst_preds, tst_reals = evaluation(model, device, tst_loader)

                    model_results['trn'] = (trn_preds, trn_reals)
                    model_results['val'] = (val_preds, val_reals)
                    model_results['tst'] = (tst_preds, tst_reals)

                    logger.info(f"(seed: {seed}) improved at epoch {best_epoch}; best loss: {best_loss}")
                else:
                    logger.info(f"(seed: {seed}) No improvement since epoch {best_epoch}; best loss: {best_loss}")


            torch.save(model_state, model_folder / f'DTA_{project_name}_seed{seed}_best.pt')

            trn_preds, trn_reals = model_results['trn']
            val_preds, val_reals = model_results['val']
            tst_preds, tst_reals = model_results['tst']

            # perform
            trn_perform = cal_perform(trn_preds, trn_reals, 'trn', task_metric)
            val_perform = cal_perform(val_preds, val_reals, 'val', task_metric)
            tst_perform = cal_perform(tst_preds, tst_reals, 'tst', task_metric)
            performs = pd.DataFrame([trn_perform, val_perform, tst_perform])
            performs['Seed'] = seed
            performs['Task'] = task_name
            performs['Project'] = project_name
            performs['Best_epoch'] = best_epoch
            performs.to_csv(model_folder / f'DTA_{project_name}_seed{seed}_best.csv', header=True, index=False)
            total_results.append(performs)

            logger.info(f'====> (seed: {seed}) best epoch: {best_epoch}; best_loss: {best_loss}')
            logger.info(f"#####" * 20)

        elif task_name == 'PPI':
            best_auc = 0
            for epoch in range(n_epochs):
                trn_preds, trn_reals, trn_loss = train(model, device, trn_loader, loss_fn, optimizer, epoch + 1, 'cls')
                val_preds, val_reals = evaluation(model, device, val_loader, 'cls')
                val_auc = roc_auc_score(val_reals.detach().cpu().numpy(), val_preds.detach().cpu().numpy())

                if val_auc > best_auc:
                    # save loss
                    best_loss = val_loss
                    best_epoch = epoch + 1
                    model_state = copy.deepcopy(model.state_dict())

                    # tst
                    tst_preds, tst_reals = evaluation(model, device, tst_loader)

                    model_results['trn'] = (trn_preds, trn_reals)
                    model_results['val'] = (val_preds, val_reals)
                    model_results['tst'] = (tst_preds, tst_reals)

                    logger.info(f"(seed: {seed}) Improved at epoch {best_epoch}; best auc: {best_auc}")
                else:
                    logger.info(f"(seed: {seed}) No improvement since epoch {best_epoch}; best auc: {best_auc}")

            torch.save(model_state, model_folder / f'PPI_{project_name}_seed{seed}_best.pt')

            trn_preds, trn_reals = model_results['trn']
            val_preds, val_reals = model_results['val']
            tst_preds, tst_reals = model_results['tst']

            # perform
            trn_perform = cal_perform(trn_preds, trn_reals, 'trn', task_metric)
            val_perform = cal_perform(val_preds, val_reals, 'val', task_metric)
            tst_perform = cal_perform(tst_preds, tst_reals, 'tst', task_metric)
            performs = pd.DataFrame([trn_perform, val_perform, tst_perform])
            performs['Seed'] = seed
            performs['Task'] = task_name
            performs['Project'] = project_name
            performs['Best_epoch'] = best_epoch
            performs.to_csv(model_folder / f'PPI_{project_name}_seed{seed}_best.csv', header=True, index=False)
            total_results.append(performs)

            logger.info(f'====> (seed: {seed}) best epoch: {best_epoch}; best_auc: {best_auc}')
            logger.info(f"#####" * 20)

        else:
            raise ValueError(f'No valid task name found: {task_name}')

    # history
    total_df = pd.concat(total_results).reset_index(drop=True)
    total_df.to_csv(output_folder / 'history.csv', index=False, header=True)

    # summary
    mean_row = []
    for group in total_df.groupby(by=['Set', 'Task', 'Project']):
        row_dict = {'Set': group[0][0], 'Task': group[0][1], 'Project': group[0][2]}
        for k, v in group[1].mean(numeric_only=True).to_dict().items():
            if k == 'Seed':
                row_dict['Seeds'] = len(group[1])
                continue
            elif k == 'Best_epoch':
                v = int(np.ceil(v))
            row_dict[k] = v
        mean_row.append(row_dict)

    summary = pd.DataFrame(mean_row)
    summary.to_csv(output_folder / f'{task_name}_{project_name}_summary.csv', index=False, header=True)

    # finish
    runtime = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    logger.info(f"Time : {runtime}")
    logger.info(f'All training jobs done')