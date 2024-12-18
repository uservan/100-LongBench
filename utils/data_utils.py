import os
import json
import uuid
from .utils import cache_dir, token
import random
import re
import math
import transformers
from datasets import load_dataset
import datasets
from datasets import Value, Sequence
import copy 

def load_data_from_json(dataset, dataset_path):
    data = []
    with open(os.path.join(dataset_path, f'{dataset}.jsonl'), "r", encoding="utf-8") as f:
        for line in f:
            l = json.loads(line)
            data.append(l)
    data = {'test':data}
    return data

## generate datasets
tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', cache_dir=cache_dir ,token=token)

def get_dataset(path='LongBench', name='narrativeqa'):
    new_dataset = list()
    if path == 'LongBench':
        dataset = load_dataset('THUDM/LongBench',name=name, split='test', cache_dir=cache_dir, streaming= True)
        for obj in dataset:
            new_dataset.append({"input": [obj['input']], "length": len(tokenizer.encode(obj['context'])), 
                            'context':obj['context'] , 'answers':[obj['answers']],
                            'dataset':name})
    if path == 'LEval':
        dataset = load_dataset('L4NLP/LEval', name=name  ,split='test', cache_dir=cache_dir, streaming= True)
        for obj in dataset:
            new_dataset.append({"input": obj['instructions'], "length": len(tokenizer.encode(obj['input'])), 
                            'context':obj['input'] , 'answers':obj['outputs'],
                            'dataset':name})
    if path == 'cnn_dailymail':
        dataset = load_dataset('abisee/cnn_dailymail', name=name  ,split='test', cache_dir=cache_dir, streaming= True)
        num = 0
        for obj in dataset:
            new_dataset.append({"input": '', "length": len(tokenizer.encode(obj['article'])), 
                            'context':obj['article'] , 'answers':[obj['highlights']],
                            'dataset':'cnn_dailymail'})
            num+=1
            if num > 1000: break
    if path == 'rag-mini-bioasq':
        passages = load_dataset('enelpol/rag-mini-bioasq', name='text-corpus' ,split='test', cache_dir=cache_dir, streaming= True)
        dataset = load_dataset('enelpol/rag-mini-bioasq', name='question-answer-passages' ,split='test', cache_dir=cache_dir, streaming= True)
        ids = set(pid for obj in dataset for pid in obj['relevant_passage_ids'])
        passages = {p['id']:p['passage'] for p in passages if p['id'] in ids}
        for obj in dataset:
            if len( obj['relevant_passage_ids']) > 1:
                context = ''
                for i, pid in enumerate(obj['relevant_passage_ids']):
                     context = context + f'Passage {i+1}:\n' + passages[pid] + '\n'
                new_dataset.append({"input": [obj['question']], "length": len(tokenizer.encode(context)), 
                                'context':context , 'answers':[obj['answer']],
                                'dataset':'rag-mini-bioasq'})
    return new_dataset

def generate_dataset_single_doc_qa_base(length=8, rows=100, **kwargs):
    # single-doc-qa
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets, choices=list(), [], []
    for name in ("qasper", "multifieldqa_en", "narrativeqa"):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list) if data['length']< length]
    if len(datasets_id) > rows: datasets_id = random.sample(datasets_id, rows)
    # collect
    results = []
    for c in datasets_id:
        d = dataset_list[c[1]]
        context = f'Passage 1:\n' + d['context'] + '\n\n'
        results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': d['context'] , 
                        'instruction': f'Answer the question related with Passage 1. ', 
                         "input": d['input'], "answers": d["answers"]})
    return results

def generate_dataset_single_doc_qa(length=8, rows=100):
    # single-doc-qa
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets, choices=list(), [], []
    for name in ("qasper", "multifieldqa_en", "narrativeqa"):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    for name in ('multidoc_qa', 'legal_contract_qa', 'financial_qa', 'natural_question', 'scientific_qa' ):
        dataset = get_dataset('LEval',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list)]
    for i in range(rows):
        l = length
        while l > 0:
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        new_datasets.append(choices)
        choices = []
    # collect
    results = []
    for choices in new_datasets:
        context = ''
        for i, c in enumerate(choices):
            context = context + f'Passage {i+1}:\n' + dataset_list[c[1]]['context'] + '\n\n'
        c_i = random.choice(range(len(choices)))
        d = dataset_list[choices[c_i][1]]
        results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': d['context'] , 
                        'instruction': f'Answer the question related with Passage {c_i+1}. ', 
                         "input": d['input'], "answers": d["answers"]})
    return results

def generate_dataset_multi_doc_qa_base(length=8, rows=100, **kwargs):
    length = length *(2**10)
    # load multi-doc datasets
    dataset_list, new_datasets = [], []
    dataset = get_dataset('rag-mini-bioasq')
    dataset_list.extend(dataset)
    for name in ("hotpotqa", "2wikimqa", "musique"):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list) if data['length']< length]
    if len(datasets_id) > rows: datasets_id = random.sample(datasets_id, rows)
    # collect
    results = []
    for c in datasets_id:
        qa = dataset_list[c[1]]
        results.append({'instruction':f'Answer the question related with these passages ' , 
                        "input": qa['input'], "answers": qa["answers"], "new_context": qa["context"],  "old_context": qa['context'], "length": len(tokenizer.encode(qa['context']))})
    return results

def generate_dataset_multi_doc_qa(length=8, rows=100):
    # multi-doc-qa
    length = length *(2**10)
    # load noise datasets
    noise_dataset_list = []
    for name in ("qasper", "multifieldqa_en", "narrativeqa"):
        dataset = get_dataset('LongBench',name)
        noise_dataset_list.extend(dataset)
    for name in ('multidoc_qa', 'legal_contract_qa', 'financial_qa', 'natural_question', 'scientific_qa' ):
        dataset = get_dataset('LEval',name)
        noise_dataset_list.extend(dataset)
    # load multi-doc datasets
    dataset_list, new_datasets = [], []
    for name in ("hotpotqa", "2wikimqa", "musique"):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    dataset = get_dataset('rag-mini-bioasq')
    dataset_list.extend(dataset)
    # generate
    datasets_id, noise_datasets_id ,choices = [(data['length'], i) for i, data in enumerate(dataset_list)], \
        [(data['length'], i) for i, data in enumerate(noise_dataset_list)],[]
    for i in range(rows):
        l = length
        datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
        choice = random.choice(datasets_id_tmp)
        choices.append(choice) 
        l = l-choice[0]     
        while l > 0:
            datasets_id_tmp = [data_id for data_id in noise_datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        new_datasets.append(choices)
        choices = []
    # collect
    results = []
    for choices in new_datasets:
        qa = dataset_list[choices[0][1]]
        split_text = [ (p, True) for p in re.split(r'Passage \d+:\n', qa['context']) if p.strip()]
        noise_text = [(noise_dataset_list[c[1]]['context'] ,False) for c in choices[1:]]
        split_text.extend(noise_text)
        random.shuffle(split_text)
        context, num = '', []
        for i, (text, flag) in enumerate(split_text):
            context = context + f'Passage {i+1}:\n' + text + '\n'
            if flag: num.append(i+1)
        results.append({'instruction':f'Answer the question related with Passage '+','.join(map(str, num))+'. ' , 
                        "input": qa['input'], "answers": qa["answers"], "new_context": context,  "old_context": qa['context'],
                        "length": len(tokenizer.encode(context))})
    return results

def generate_dataset_single_doc_sum_base(length=8, rows=100, **kwargs):
   # single-doc-sum
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets, choices=list(), [], []
    for name in ("gov_report", "qmsum"): # "multi_news"
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    for name in ( 'patent_summ','tv_show_summ','review_summ',  'meeting_summ' ): # 'news_summ',
        dataset = get_dataset('LEval',name)
        dataset_list.extend(dataset)
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list) if data['length']< length]
    if len(datasets_id) > rows: datasets_id = random.sample(datasets_id, rows)
    # collect
    results = []
    for c in datasets_id:
        d = dataset_list[c[1]]
        context = f'Passage 1:\n' + d['context'] + '\n\n'
        instruction = f'Write a one-page summary of Passage 1 into a few short sentences'
        results.append({ 'instruction': '',
                         "input": [f'{instruction}: {inp}' if len(inp) > 0 else f'{instruction}.' for inp in d['input']], 
                         "answers": d["answers"], "new_context": context, 'old_context':d['context'],
                         "length": len(tokenizer.encode(context))})
    return results

def generate_dataset_single_doc_sum(length=8, rows=100):
    # single-doc-sum
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets, choices=list(), [], []
    for name in ("gov_report", "qmsum"): # "multi_news"
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    for name in ('patent_summ','tv_show_summ','review_summ',  'meeting_summ' ): # 'news_summ',
        dataset = get_dataset('LEval',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list)]
    for i in range(rows):
        l = length
        while l > 0:
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        new_datasets.append(choices)
        choices = []
    # collect
    results = []
    for choices in new_datasets:
        context = ''
        for i, c in enumerate(choices):
            context = context + f'Passage {i+1}:\n' + dataset_list[c[1]]['context'] + '\n\n'
        c_i = random.choice(range(len(choices)))
        d = dataset_list[choices[c_i][1]]
        instruction = f'Write a one-page summary of Passage {c_i+1} into a few short sentences'
        results.append({ 'instruction': '',
                         "input": [f'{instruction}: {inp}' if len(inp) > 0 else f'{instruction}.' for inp in d['input']], 
                         "answers": d["answers"], "new_context": context, 'old_context':d['context'],
                         "length": len(tokenizer.encode(context))})
    return results

def generate_dataset_multi_doc_sum_base(length=8, rows=100, **kwargs):
    # multi-doc-sum
    length = length *(2**10)
    # load multi-doc datasets
    dataset_list, new_datasets = [], []
    for name in ("multi_news_e", ):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list) if data['length']< length]
    if len(datasets_id) > rows: datasets_id = random.sample(datasets_id, rows)
    # collect
    results = []
    for c in datasets_id:
        qa = dataset_list[c[1]]
        split_text = [ (p, True) for p in re.split(r'Passage \d+:\n', qa['context']) if p.strip()]
        random.shuffle(split_text)
        context, num = '', []
        for i, (text, flag) in enumerate(split_text):
            context = context + f'Passage {i+1}:\n' + text + '\n'
            if flag: num.append(f'Passage {i+1}')
        instruction = 'Combine and summarize the main ideas from the selected relevant passages into one cohesive summary: '+','.join(map(str, num))
        results.append({'instruction':'' , 
                        "input": qa['input'], 
                        "input": [f'{instruction}: {inp}' if len(inp) > 0 else f'{instruction}.' for inp in qa['input']], 
                        "answers": qa["answers"], "new_context": context,  "old_context": qa['context'],
                        "length": len(tokenizer.encode(context))})
    return results

def generate_dataset_multi_doc_sum(length=8, rows=100):
    # multi-doc-sum
    length = length *(2**10)
    # load noise datasets
    noise_dataset_list = []
    for name in ("gov_report", "qmsum"):
        dataset = get_dataset('LongBench',name)
        noise_dataset_list.extend(dataset)
    for name in ('patent_summ','tv_show_summ','review_summ',  'meeting_summ' ):
        dataset = get_dataset('LEval',name)
        noise_dataset_list.extend(dataset)
    # load multi-doc datasets
    dataset_list, new_datasets = [], []
    for name in ("multi_news_e", ):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id, noise_datasets_id ,choices = [(data['length'], i) for i, data in enumerate(dataset_list)], \
        [(data['length'], i) for i, data in enumerate(noise_dataset_list)],[]
    for i in range(rows):
        l = length
        datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
        choice = random.choice(datasets_id_tmp)
        choices.append(choice) 
        l = l-choice[0]     
        while l > 0:
            datasets_id_tmp = [data_id for data_id in noise_datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        new_datasets.append(choices)
        choices = []
    # collect
    results = []
    for choices in new_datasets:
        qa = dataset_list[choices[0][1]]
        split_text = [ (p, True) for p in re.split(r'Passage \d+:\n', qa['context']) if p.strip()]
        noise_text = [(noise_dataset_list[c[1]]['context'] ,False) for c in choices[1:]]
        split_text.extend(noise_text)
        random.shuffle(split_text)
        context, num = '', []
        for i, (text, flag) in enumerate(split_text):
            context = context + f'Passage {i+1}:\n' + text + '\n'
            if flag: num.append(f'Passage {i+1}')
        instruction = 'Combine and summarize the main ideas from the selected relevant passages into one cohesive summary: '+','.join(map(str, num))
        results.append({'instruction':'' , 
                        "input": qa['input'], 
                        "input": [f'{instruction}: {inp}' if len(inp) > 0 else f'{instruction}.' for inp in qa['input']], 
                        "answers": qa["answers"], "new_context": context,  "old_context": qa['context'],
                        "length": len(tokenizer.encode(context))})
    return results

def generate_dataset_kv_retrieval(length=8, rows=100, kv_num=3):
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets=[], []
    for name in ("qasper", "multifieldqa_en", "narrativeqa"):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    for name in ('multidoc_qa', 'legal_contract_qa', 'financial_qa', 'natural_question', 'scientific_qa' ):
        dataset = get_dataset('LEval',name)
        dataset_list.extend(dataset)
    dataset = get_dataset('cnn_dailymail', '1.0.0')
    dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list)]
    while len(new_datasets) < rows:
        choices = []
        l = length
        while l > 0:
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        if len(choices)<kv_num+1: continue
        new_datasets.append(choices)
    # collect
    results = []
    for choices in new_datasets:
        context, keys, position = '', [str(uuid.uuid4()) for _ in range(4)], random.sample(range(len(choices)), kv_num)
        key_value = dict()
        for i, p in enumerate(position):
            key_value[p] = f'\n\nThe value of {keys[i]} is {keys[i+1]}.\n\n'
        for i, c in enumerate(choices):
            if i in key_value.keys():
                context = context + key_value[i]
            context = context + f'Passage {i+1}: ' + dataset_list[c[1]]['context'].lstrip('\n').lstrip(' ') + '\n'
        # question_key = keys[:4]
        # random.shuffle(question_key)
        # results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': '' , 
        #                 'instruction': '', 
        #                  "input": [f'Now, the key is {keys[0]}, and what is the value of this key?'+
        #                            f'\n1. {question_key[0]}\n2. {question_key[1]}\n3. {question_key[2]}\n4. {question_key[3]}\n'+
        #                            'Please provide your answer as a single number (1, 2, 3, or 4) without any explanation.']
        #                            ,"answers":[[question_key.index(keys[4])+1,keys[4]]]})
        results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': '' , 
                        'instruction': '', 
                         "input": [f'Answer: the value of {keys[0]} is']
                                   ,"answers":[keys[1]]})
    return results

def generate_dataset_counting_stars(length=8, rows=100, test_type='Acquisition'):
    # single-doc-qa
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets, choices=list(), [], []
    for name in ("qasper", "multifieldqa_en", "narrativeqa"):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    for name in ('multidoc_qa', 'legal_contract_qa', 'financial_qa', 'natural_question', 'scientific_qa' ):
        dataset = get_dataset('LEval',name)
        dataset_list.extend(dataset)
    dataset = get_dataset('cnn_dailymail', '1.0.0')
    dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list)]
    while len(new_datasets) < rows:
        choices = []
        l = length
        while l > 0:
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        if len(choices)<4: continue
        new_datasets.append(choices)
    # collect
    results = []
    def modify(my_list):
        index1, star = random.sample(range(len(my_list)), 1), random.randint(1, 100)
        my_list[index1[0]] = star
        return my_list
    def exchange(my_list):
        index1, index2 = random.sample(range(len(my_list)), 2)
        my_list[index1], my_list[index2] = my_list[index2], my_list[index1]
        return my_list
    for choices in new_datasets:
        context, whole_stars = '', 0
        answers, noise_answer, positions = [],[],random.sample(range(len(choices)), 4)
        for i, c in enumerate(choices):
            if i in positions:
                a_stars, r_stars = random.randint(1, 100), random.randint(1, 100)
                if test_type == 'Acquisition':
                    single_star = f"\n\nThe little penguin counted {a_stars} ★\n\n"
                if test_type == 'Reasoning':
                    single_star = f"\n\nThe little penguin counted {r_stars} ★, but found that a mistake had been made, so the counting was done again, and this time {a_stars} ★ was counted correctly.\n\n"
                whole_stars = whole_stars+a_stars
                context = context + single_star
                answers.append(a_stars)
                noise_answer.append(r_stars)
            context = context + f'Passage {i+1}:\n' + dataset_list[c[1]]['context'] + '\n'
        question_key = []
        if test_type == 'Acquisition':
            question = f"On this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. Please help the little penguin collect the number of ★, for example: [x, x, x,...]. The summation is not required, and the numbers in [x, x, x,...]. represent the counted number of ★ by the little penguin.\n"
            question_key = [exchange(answers[:]), modify(answers[:]), answers[:], exchange(answers[:])]
        if test_type == 'Reasoning':
            question = f"On this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. Please help the little penguin collect the correct number of ★, for example: [x, x, x,...]. The summation is not required, and the numbers in [x, x, x,...]. represent the correctly counted number of ★ by the little penguin.\n"
            question_key = [exchange(answers[:]), modify(answers[:]), answers[:], noise_answer[:], exchange(noise_answer[:]), modify(noise_answer[:])]
        question=question+'Question: Which of the following is the correct number of ★ the little penguin collect?\n'
        random.shuffle(question_key)
        question_key = ['unanswerable']+question_key
        for q_i, q in enumerate(question_key):
            question=question+f'{q_i+1}. {q}\n'
        question = question +'Please provide your answer as a single number (1, 2, 3, 4 ...) without any explanation.'
        results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': '' , 
                        'instruction': '',  "input": [question], "answers": [[question_key.index(answers)+1, f'{answers}']]})
        
        # if test_type == 'Acquisition':
        #     question = "On this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. Please help the little penguin collect the number of ★, for example: {\"little_penguin\": [x, x, x,...]}. The summation is not required, and the numbers in [x, x, x,...] represent the counted number of ★ by the little penguin."
        # if test_type == 'Reasoning':
        #     question = "On this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. Please help the little penguin collect the correct number of ★, for example: {\"little_penguin\": [x, x, x,...]}. The summation is not required, and the numbers in [x, x, x,...] represent the correctly counted number of ★ by the little penguin."
        # question = question +'Only output the results in JSON format without any explanation.'
        # results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': '' , 
        #                 'instruction': '',  "input": [question], "answers": [f'{answers}']})
    return results

def generate_dataset_passage_count(length=8, rows=100):
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets=[], []
    def split(data):
        l = len(tokenizer.encode(data['context']))
        if l <=length: dataset_list.append(data)
    for name in ("qasper", "multifieldqa_en", "narrativeqa"):
        dataset = get_dataset('LongBench',name)
        for data in dataset: split(data)
    for name in ('multidoc_qa', 'legal_contract_qa', 'financial_qa', 'natural_question', 'scientific_qa'):
        dataset = get_dataset('LEval',name)
        for data in dataset: split(data)
    dataset = get_dataset('cnn_dailymail', '1.0.0')
    for data in dataset: split(data)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list)]
    while len(new_datasets) < rows:
        choices, l = [], length
        while l > 0:
            passage_num = random.sample(range(3), 1)[0]+1
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]*passage_num<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append((choice, passage_num))
            l = l-choice[0]*passage_num
        new_datasets.append(choices)
    # collect
    results = []
    for choices in new_datasets:
        passages_num = len(choices)
        choices = [c for (c, num) in choices for i in range(num)]
        random.shuffle(choices)
        context = ''
        for i, c in enumerate(choices):
            context = context + f'Paragraph {i+1}: ' + dataset_list[c[1]]['context'].lstrip('\n').lstrip(' ') + '\n\n'
        results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': '' , 
                        'instruction': '',  "input":[''], "answers": [f'{passages_num}']})
    return results

def generate_dataset_passage_retrieval(length=8, rows=100):
    # single-doc-sum
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets, choices=list(), [], []
    for name in ("gov_report", "qmsum"): # "multi_news"
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    for name in ( 'patent_summ','tv_show_summ','review_summ',  'meeting_summ' ): # 'news_summ',
        dataset = get_dataset('LEval',name)
        dataset_list.extend(dataset)
    dataset = get_dataset('cnn_dailymail', '1.0.0')
    dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list)]
    for i in range(rows):
        l = length
        while l > 0:
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        new_datasets.append(choices)
        choices = []
    # collect
    results = []
    for choices in new_datasets:
        context = ''
        for i, c in enumerate(choices):
            context = context + f'Passage {i+1}: ' + dataset_list[c[1]]['context'].lstrip('\n').lstrip(' ') + '\n\n'
        c_i = random.choice(range(len(choices)))
        d = dataset_list[choices[c_i][1]]
        results.append({ 'instruction': '',
                         "input": [d["answers"][0]], "answers": [f'{c_i+1}'] , "new_context": context, 
                         'old_context':d['context'], "length": len(tokenizer.encode(context))})
    return results
   
def generate_dataset_synthetic_base(max_length=8, rows=100, func=None, **kwargs):
    lens = list(range(2, max_length, 2))
    each_rows = math.floor(rows/len(lens))
    results = []
    for i in lens:
        results.extend(func(i, each_rows))
    return results

