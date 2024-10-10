import torch
import logging
import numpy as np
import torch_geometric as pyg
logger = logging.getLogger(__name__)

def set_log(path_output, log_message):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(path_output / log_message),
            logging.StreamHandler()
        ]
    )


def set_random_seeds(seed: int):
    pyg.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"set seed: {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def cal_perform(pred, real, dt_name, lfs):
    result = {'Set': dt_name}
    for lfn, lf in lfs.items():
        if lfn in ['MSE', 'MAE', 'Huber']:
            result[lfn] = float(f"{lf(pred, real):.3f}")
        else:
            pred = pred.detach().cpu().numpy()
            real = real.detach().cpu().numpy()
            result[lfn] = float(f"{lf(real, pred):.3f}")
    return result