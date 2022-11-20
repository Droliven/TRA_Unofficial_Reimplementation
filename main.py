import logging
import os
import torch
import inspect
import numpy as np
import random

from utils.logging_help import Log

# ======================================================================================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_torch(seed=3450):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

# ======================================================================================================================

import argparse
import yaml
from getpass import getuser

from runs.run import Run_TRAModel

parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--config_path', type=str, default='./configs/config_alstm.yaml', help="")
parser.add_argument('--config_path', type=str, default='./configs/config_alstm_tra_init.yaml', help="")
# parser.add_argument('--config_path', type=str, default='./configs/config_alstm_tra.yaml', help="")

# parser.add_argument('--config_path', type=str, default='./configs/config_transformer.yaml', help="")
# parser.add_argument('--config_path', type=str, default='./configs/config_transformer_tra_init.yaml', help="")
# parser.add_argument('--config_path', type=str, default='./configs/config_transformer_tra.yaml', help="")

args = parser.parse_args()

with open(args.config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


seed = cfg["model"]["seed"]
seed_torch(seed)

logger = Log(log_file_name=f'{getuser()}', log_level=logging.DEBUG, log_dir=cfg["model"]["logdir"]).returnLogger()
logging.info(f"cuda idx: {os.environ['CUDA_VISIBLE_DEVICES']}")


r = Run_TRAModel(logger=logger, device="cuda", is_debug=cfg["dataset"]["is_debug"], dataset_cfg=cfg["dataset"], **cfg["model"])
r.fit()

