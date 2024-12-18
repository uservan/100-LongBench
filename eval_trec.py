import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.model_utils import load_model
from utils.utils import set_global_path, logger
from utils.metrics import (
    qa_f1_score,
    rouge_score,
    acc_score,
    kv_retrieval_score,
    qa_model_score,
    sum_model_score
)

dataset2metric = {
    'qa':qa_f1_score, # qa_f1_score, qa_model_score
    'sum':rouge_score, # rouge_score, sum_model_score
    "passage_count": acc_score,
    "passage_retrieval": acc_score,
    "counting_stars": acc_score,
    "kv_retrieval": kv_retrieval_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-3-8B-Instruct_96.0')
    parser.add_argument('--qa_filter', type=float, default=0.5)  # 0.5
    parser.add_argument('--metric_model', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--pred_path', type=str, default='pred_new2')
    parser.add_argument('--dataset_path', type=str, default='data/open_model')
    return parser.parse_args(args)

def scorer(dataset, predictions, answers, qas, null_preds, qa_filter, all_classes):
    total_score = 0.
    datasets_type = ["qa",'sum','passage_count','passage_retrieval', 'counting_stars', 'kv_retrieval']
    for d in datasets_type: 
        if d in dataset: dataset=d
    num=0
    for i, prediction in enumerate(tqdm(predictions, desc="Evaluating")):
        ground_truths, qa, null_pred, score=answers[i], None, None, 0.
        if len(qas)>0: 
            qa, null_pred = qas[i], null_preds[i]
            if qa_filter>0:
                null_pred = null_pred.lstrip('\n').split('\n')[0]
                if 'unanswerable' not in null_pred.lower(): 
                    for ground_truth in ground_truths:
                        score = max(score, dataset2metric[dataset](null_pred, ground_truth, qa=qa, model=model, all_classes=all_classes))
                    if score>=qa_filter: continue
        score=0
        if dataset in ['qa','passage_count','passage_retrieval','counting_stars','kv_retrieval', 'sum']:
            prediction = prediction.lstrip('\n').split('\n')[0]
        if len(prediction) != 0:
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](prediction, ground_truth, qa=qa, model=model, all_classes=all_classes))
        total_score += score
        num+=1
    return round(100 * total_score / num, 2)

if __name__ == '__main__':
    args = parse_args()
    if args.metric_model: model=load_model(args.metric_model, temperature=0.1)
    path = set_global_path(f"{args.pred_path}/{args.model}/")
    out_path = set_global_path(f"{args.pred_path}/{args.model}/result.json")
    all_scores = dict()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            file_content = f.read()    
            all_scores = json.loads(file_content) 
    all_files = sorted(os.listdir(path))
    for filename in all_files:
        if not filename.endswith("jsonl"): continue
        predictions, answers, qa, null_preds, lengths = [], [], [], [], []
        dataset = filename.split('.')[0]
        if dataset in all_scores.keys(): continue
        logger("Evaluating on:", filename)
        flag = True
        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if "pred" not in data.keys(): 
                    flag=False
                    break
                predictions.append(data["pred"])
                if type(data["answers"]) != list: 
                    data["answers"] = [data["answers"]]
                answers.append(data["answers"])
                all_classes = filename
                if "length" in data:
                    lengths.append(data["length"])
                if 'qa' in dataset:
                    qa.append(data['question'])
                    null_preds.append(data['null_pred'])
        if flag:
            score = scorer(dataset, predictions, answers, qa, null_preds, args.qa_filter ,all_classes)
            all_scores[dataset]=score
            with open(out_path, "w") as f:
                json.dump(all_scores, f, ensure_ascii=False, indent=4)
