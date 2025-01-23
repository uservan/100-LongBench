# 💯-LongBench

### Generate Evaluation Datasets
- Each task generates datasets of different lengths, such as 8k, 16k, 32k, 64k, 128k, and 256k, with each dataset containing 100 data samples.
    ```shell
    python generate_dataset.py --rows 100 --lengths 8,16,32,64,128,256 --save_path data/open_model
    ```
### Get Model Prediction
- The ./config/model2path.json file lists the currently available models. 
    - To add a new model, simply include the corresponding Hugging Face model path or the local file path.
    - for example, if you wanna add Llama-3.1-8B-Instruct, you could add as follows
        ```json
        {   
            ...
            "llama-3.1-8B-Instruct":"meta-llama/Llama-3.1-8B-Instruct"
            ...
        }
        ```
- run the code to get predictions
    ```shell
    python pred.py --model llama-3-8B-Instruct --pred_path preds/pred_open_model --dataset_path data/test
    ```

### Evaluation
- use the metric which is same as LongBench
    ```shell
    python eval_trec.py --pred_path preds/pred_open_model --model llama-3.1-8B-Instruct --original_metric True
    ```
- use model-based metric without qa_filter
    ```shell
    python eval_trec.py --pred_path preds/pred_open_model --model llama-3.1-8B-Instruct --original_metric False --metric_model gpt-4o-mini-2024-07-18
    ```
- use model-based metric with qa_filter
    - Filter the model’s answer scores for a question without context. If the score exceeds qa_filter, the current data is not included in the results.
    ```shell
    python eval_trec.py --pred_path preds/pred_open_model --model llama-3.1-8B-Instruct --original_metric False --metric_model gpt-4o-mini-2024-07-18 --qa_filter 0.5
    ```
### Analysis
- The code for detailed result analysis can be found in analysis/result_analysis.ipynb.
- The validation section mentioned in the paper can be referred to in analysis/verification.ipynb.
