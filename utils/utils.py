import os
import numpy as np
import random
import torch
import time
from huggingface_hub import login

project_dir = './'
cache_dir = './cache'
token= 'your token'
api_keys = {
    'openai': 'your api key'
}

login(token=token)

def set_global_path(path):
    return os.path.join(project_dir, path)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def logger(text):
    print(text)