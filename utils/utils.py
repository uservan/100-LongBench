import os
import numpy as np
import random
import torch
import time
from huggingface_hub import login

cache_dir = '/users/PDS0352/wyang107/project/LCEG/model_cache'
token= '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'
api_keys = {
    'openai': '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'
}


login(token=token)

def set_global_path(path):
    return os.path.join('/users/PDS0352/wyang107/project/100-LongBench', path)

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