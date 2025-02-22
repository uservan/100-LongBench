import os
import sys
import json
from tqdm import tqdm
from utils.utils import set_global_path, seed_everything, logger
from utils.data_utils import *
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
sys.path.append(models_dir)


func = {
        'single_doc_qa':[generate_dataset_single_doc_qa,generate_dataset_single_doc_qa_base],
        'multi_doc_qa': [generate_dataset_multi_doc_qa,generate_dataset_multi_doc_qa_base],
        'single_doc_sum':[generate_dataset_single_doc_sum, generate_dataset_single_doc_sum_base],
        'multi_doc_sum': [generate_dataset_multi_doc_sum, generate_dataset_multi_doc_sum_base],
        'passage_retrieval':[generate_dataset_passage_retrieval,generate_dataset_synthetic_base],
        'passage_count':[generate_dataset_passage_count, generate_dataset_synthetic_base],
        'kv_retrieval': [generate_dataset_kv_retrieval, generate_dataset_synthetic_base], 
        'counting_stars':[generate_dataset_counting_stars, generate_dataset_synthetic_base]
}

seed_everything(42)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=100, help='The number of each dataset')
    parser.add_argument('--lengths', type=str, default='8,16,32,64,128,256', help='Generated length, where 8 represents 8*1024')
    parser.add_argument('--save_path', type=str, default='data/open_model')
    return parser.parse_args(args)

args = parse_args()
out_path_tmp = set_global_path(args.save_path)
lengths = [int(i) for i in args.lengths.split(',')]
# generate base
rows, length = args.rows, lengths[0] # 
if not os.path.exists(out_path_tmp): os.makedirs(out_path_tmp)
for key in func.keys():
    result = func[key][1](length, rows, func=func[key][0])
    with open(os.path.join(out_path_tmp, f'{key}_0.jsonl'), "w", encoding="utf-8") as f:
        for pred in result:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')
# generate each task in different lengths
rows, lengths = args.rows, lengths
if not os.path.exists(out_path_tmp): os.makedirs(out_path_tmp)
for length in tqdm(lengths):
    for key in func.keys():
        result = func[key][0](length, rows)
        with open(os.path.join(out_path_tmp, f'{key}_{length}.jsonl'), "w", encoding="utf-8") as f:
            for pred in result:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')


