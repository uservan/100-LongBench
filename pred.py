import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import argparse
import sys
from utils.utils import seed_everything, set_global_path, logger, cache_dir
from utils.model_utils import load_model, LLM
from utils.data_utils import load_data_from_json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
sys.path.append(models_dir)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-3-8B-Instruct')
    parser.add_argument('--pred_path', type=str, default='preds/pred_ratio')
    parser.add_argument('--dataset_path', type=str, default='data/open_model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratio', type=float, default=1.0)
    return parser.parse_args(args)

def get_pred(model:LLM, data, max_gen, prompt_format, preds=[] ,out_path=''):
    data = data['test']
    k = 0
    for json_obj in tqdm(data):
        context_length = json_obj["length"]
        try:
            for i, inp in enumerate(json_obj['input']):
                flag=False
                if k < len(preds):
                    pred, flag = preds[k], True
                    k = k+1
                    if len(pred['answers']) != 0: continue
                else: k = k+1
                obj = {'context':json_obj['new_context'], 'input':inp, "length":json_obj["length"],
                        'answers':json_obj['answers'][i], 'instruction':json_obj['instruction']}
                prompt = prompt_format[0].format(**obj)
                pred = model.generate(prompt, max_gen)
                null_pred = ''
                if len(prompt_format)>1:
                    obj = {'context':'\nThere are no passages.\n', 'input':inp, "length":json_obj["length"],
                        'answers':json_obj['answers'][i], 'instruction':json_obj['instruction']}
                    prompt = prompt_format[1].format(**obj)
                    null_pred = model.generate(prompt, max_gen)
                    pred = {"pred": pred, "answers": obj["answers"], "length": context_length, 'null_pred':null_pred, 'question':inp}
                else:
                    pred = {"pred": pred, "answers": obj["answers"], "length": context_length, 'null_pred':null_pred}
                if flag: preds[k-1] = pred 
                else: preds.append(pred)
        except Exception as e:
            logger(f"Exception occurred: {e}")
            pred = {"answers":'', "length": json_obj["length"]}
            if flag: preds[k-1] = pred
            else: preds.append(pred)
        save_preds(out_path, preds)
    return preds

def save_preds(out_path, preds):
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')

def read_preds(out_path):
    preds=[]
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                l = json.loads(line)
                preds.append(l)
    return preds

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    
    pred_path = set_global_path(args.pred_path) #  set_global_path("pred_new2")
    dataset_path = set_global_path(args.dataset_path) # set_global_path("data/test")
    if not os.path.exists(pred_path): os.makedirs(pred_path)
    model_name = args.model
    logger(model_name)
    # define your model
    model = load_model(model_name, ratio=args.ratio)
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open(set_global_path("./config/dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(open(set_global_path("./config/dataset2maxlen.json"), "r"))
    
    file_names = [p.split('.')[0] for p in os.listdir(dataset_path)]
    save_dir = f"{pred_path}/{model_name}"
    if args.ratio>0: save_dir = f"{pred_path}/{model_name}_{args.ratio}"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    for dataset in file_names:
        logger(f'testing: {dataset}')
        # data = load_dataset(os.path.join(dataset_path, f'{dataset}.jsonl'))
        # data = load_dataset(dataset_path, dataset, split='test', cache_dir=cache_dir)
        # data = load_dataset('json', data_files={'test': os.path.join(dataset_path, f'{dataset}.jsonl')})
        data = load_data_from_json(dataset, dataset_path)
        out_path = f"{save_dir}/{dataset}.jsonl"
        key = '_'.join(dataset.split('_')[:-1])
        if 'qa' in key: prompt_format = [dataset2prompt[key], dataset2prompt['qa_old']]
        else: prompt_format = [dataset2prompt[key]]
        max_gen = dataset2maxlen[key]
        preds = read_preds(out_path)
        preds = get_pred(model, data, max_gen, prompt_format, preds, out_path)