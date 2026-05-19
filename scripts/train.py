import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from src.config import load_config
import random
import numpy as np
import torch
from src.data.utils import *
import logging
import warnings
from src.piap_trainer import PIAPMambaTrainer

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training_config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    if config.train.device == "cuda" and not torch.cuda.is_available():
        config.train.device = "cpu"
    
    seed_everything(config.train.seed)
    
    trainer = PIAPMambaTrainer(config.data.data_loc, config)
    trainer.train()
    
    
