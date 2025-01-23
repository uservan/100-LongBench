import os
import numpy as np
import random
import torch
import time
import pandas as pd
from collections import defaultdict
from huggingface_hub import login
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


change_names = {
    'multi_news': 'Multi-News',
    'gov_report': 'Gov Report',
    'qasper': 'Qasper',
    'counting_stars': 'Counting Stars',
    'kv_retrieval': 'KV Retrieval',
    'multi_doc_qa': 'Multi-Doc QA',
    'multi_doc_sum': 'Multi-Doc Sum',
    'passage_count': 'Passage Count',
    'passage_retrieval': 'Passage Retrieval',
    'single_doc_qa': 'Single-Doc QA',
    'single_doc_sum': 'Single-Doc Sum',
    'llama-3.1-8B':'Llama3.1-8B-Instruct',
    'llama-3.1-70B':'Llama3.1-70B-Instruct',
     'llama-3.2-1B':'Llama3.2-1B-Instruct',
     'llama-3.2-3B':'Llama3.2-3B-Instruct',
     'phi-3-128k-mini1':'Phi3-mini-Instruct',
     'phi-3-128k-medium2':'Phi3-medium-Instruct',
     'Qwen2.5-7B':'Qwen2.5-7B-Instruct',
     'Qwen2.5-14B':'Qwen2.5-14B-Instruct',
}
def change_name(name):
    if name in change_names.keys():
        return change_names[name]
    return name

def get_model_avg(draw_tasks, lengths=[0,8,16,24,31,48,64]):
    model_average_results = dict() # [0, 8, 16, 32, 64, 128]
    for task, scores in draw_tasks.items():
        # print(task)
        for model, s_l in scores.items():
            # print(model)
            old_s_l = {length: s for length, s in s_l}
            # print(old_s_l)
            if model not in model_average_results.keys():
                new_s_l = [0] * len(lengths) 
            else:
                new_s_l = model_average_results[model]
            for i, length in enumerate(lengths):
                new_s_l[i] = new_s_l[i]+old_s_l[length]
            model_average_results[model] = new_s_l
    for model, socres in model_average_results.items():
        model_average_results[model] = [(lengths[i] ,s/len(draw_tasks.keys())) for i, s in enumerate(model_average_results[model])]
    return model_average_results

def get_results_per_task(results, lengths=[0,8,16,24,31,48,64]):
    draw_tasks = defaultdict(dict)
    for model in results.keys():
        draw_things = defaultdict(list)
        for dataset,score in results[model].items():
            name = dataset.split('_')
            name, x = '_'.join(name[:-1]), int(name[-1])
            draw_things[name].append((x, score))
        draw_things = {key: draw_things[key] for key in sorted(draw_things)}
        for dataset, scores in draw_things.items():
            draw_tasks[dataset][model] = [s for s in scores if s[0] in lengths]
    return draw_tasks

def rank(performance_dict, all_steps=[0,8,16,24,31,48,64], show=True):
    original_performances, new_performances = {model:[0]*len(all_steps) for model in performance_dict}, {model:[0]*len(all_steps) for model in performance_dict}
    
    for model, scores in performance_dict.items():
        score_dict = {s: score for s, score in scores}
        base_performance = score_dict.get(0, None) 
        for i, step in enumerate(all_steps):
            current_performance = score_dict.get(step, None)
            new_performance = 100 * (current_performance - base_performance) / base_performance
            new_performance = round(new_performance, 2)
            original_performances[model][i] = current_performance
            new_performances[model][i] = new_performance
        # compute average perfomance
        lst = original_performances[model][1:]
        s = round(sum(lst) / len(lst), 2)
        original_performances[model].append(s)
        lst = new_performances[model][1:]
        s = round(sum(lst) / len(lst), 2)
        new_performances[model].append(s)

    if show:
        print('original_metrics')
        df = pd.DataFrame(original_performances, index=all_steps+[-1]).T
        print(df)
        print('\n')
        print('original_ranking')
        rank_df = df.rank(ascending=False, method='min').astype(int)
        print(rank_df)
        print('\n')
        print('new_metrics')
        df = pd.DataFrame(new_performances, index=all_steps+[-1]).T
        print(df)
        print('\n')
        print('new_ranking')
        rank_df = df.rank(ascending=False, method='min').astype(int)
        print(rank_df)
        print('\n')
    return original_performances, new_performances

def draw_avg_line(original_performances, lengths, save=False):
    fig, axes = plt.subplots(1, 1, figsize=(8, 5), sharex=True)
    colors, models = plt.cm.get_cmap('tab10', len(original_performances.keys())), list()
    x_t, k = None,None
    for m_i, (model, ranks) in enumerate(original_performances.items()):
        if model not in models: models.append(model)
        x, y = [i for i, s_i in enumerate(lengths)], original_performances[model][:-1]
        if k is None or x_t is None or len(k) < len(y):
            x_t = x
            k = ['0-8k']+[f'{s_i}k' for s_i in lengths[1:]]
        axes.plot(x, [original_performances[model][-1]]*len(x), color=colors(m_i), linestyle='--')
        axes.plot(x, y, label=change_name(model),  marker='o', linestyle='-', color=colors(m_i), markersize=15)
        axes.set_ylabel('Score')
        axes.set_xticks(x_t, k)
        axes.grid(True)

    handles = [plt.Line2D([0], [0], color=colors(i), linestyle='-', linewidth=2, label=task) for i, task in enumerate(models)]
    fig.legend(handles, [change_name(i) for i in list(models)] , loc='upper center', ncol=3, fontsize=15, bbox_to_anchor=(0.5, 1.12))

    # 显示图形
    if save: plt.savefig("results_original.pdf", format="pdf", dpi=600, bbox_inches="tight")
    plt.tight_layout()  # 自动调整子图间的间距
    plt.show()