# FinTagging

## create your project fork from https://github.com/The-FinAI/FinBen

## github repo clone
```bash
git clone https://github.com/Yan2266336/FinBen.git --recursive
```

## set environment
```bash
conda create -n finben python=3.12
conda activate finben
```

```bash
cd FinBen/finlm_eval/
pip install -e .
pip install -e .[vllm]
```

## login to your huggingface
```bash
export HF_TOKEN="your_hf_token"
```
## verify it
```bash
echo $HF_TOKEN
```

## model evaluation
```bash
cd FinBen/
```
### for GPT model
```bash
lm_eval --model openai-chat-completions\
        --model_args "model=gpt-4o" \
        --tasks GRQAGen \
        --output_path results \
        --use_cache ./cache \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ./tasks
```

### for small model
```bash
lm_eval --model hf \
        --model_args "pretrained=Qwen/Qwen2.5-0.5B" \
        --tasks regAbbreviation \
        --num_fewshot 0 \
        --device cuda:1 \
        --batch_size 8 \
        --output_path results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-reAbbr-0shot-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ./tasks
```

### for large model
```bash
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
```
```bash
lm_eval --model vllm \
        --model_args "pretrained=google/gemma-2-27b-it,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" \
        --tasks GRQA \
        --batch_size auto \
        --output_path results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ./tasks
        
lm_eval --model vllm \
        --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_length=8192" \
        --tasks GRFNS2023 \
        --batch_size auto \
        --output_path results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ./tasks
```
***results will be saved to FinBen/results/, you could add it into .gitignore***

- **0-shot setting:** Use `num_fewshot=0` and `lm-eval-results-gr-0shot` as the results repository.
- **5-shot setting:** Use `num_fewshot=5` and `lm-eval-results-gr-5shot` as the results repository.
- **Base models:** Remove `apply_chat_template`.
- **Instruction models:** Use `apply_chat_template`.

## add new task
```bash
cd FinBen/tasks/your_project_folder/ # create yaml file for new task
```

### For example
```bash
cd FinBen/tasks/fortune/ # create yaml file for new task
```
- **lm-evaluation-harness/docs/task_guide.md** # Good Reference Tasks


## if push to leaderboard
```bash
vim FinBen/aggregate.py  #change to your project huggingface repos in line 415 and 42
```

## new model
```bash
vim FinBen/aggregate.py # add new model configuration to MODEL_DICT

python aggregate.py
```
**https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=qwen2.5-1.5b-instruct**
**https://mot.isitopen.ai/**

## new task
```bash
vim FinBen/aggregate.py ##add new task to METRIC_DICT # for classification task, change 1.0 / 6.0 to your baseline

python aggregate.py
```
### huggingface leaderboard part
#### backend/app/services/leaderboard.py
#### frontend/src/pages/LeaderboardPage/components/Leaderboard/utils/columnUtils.js
#### frontend/src/pages/LeaderboardPage/components/Leaderboard/constants/tooltips.js
#### frontend/src/pages/LeaderboardPage/components/Leaderboard/constants/defaults.js
#### frontend/src/pages/QuotePage/QuotePage.js



