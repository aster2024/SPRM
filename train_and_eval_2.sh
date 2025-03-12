#!/bin/bash

declare -A model_name_map=(
    ["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.2"
    ["llama3-8b"]="meta-llama/Llama-3.1-8B-Instruct"
    ["llama3-70b"]="meta-llama/Llama-3.1-70B-Instruct"
)

declare -A dataset_file_map=(
    ["mistral-7b"]="testset/math-Mistral-7B-Instruct-v0.2-64.json"
    ["llama3-8b"]="testset/math-llama3.1-8b-inst-64.json"
    ["llama3-70b"]="testset/math-Meta-Llama-3.1-70B-Instruct-64.json"
)

declare -A max_samples_map=(
    ["mistral-7b"]=300
    ["llama3-8b"]=500
    ["llama3-70b"]=100
)

#CUDA_VISIBLE_DEVICES=7 python train/train_combined_prm.py --model_name meta-llama/Llama-3.1-8B-Instruct --max_samples 500 --method hinge --train_mode prm --input_file data/ultrainteract_math_rollout_with_rewards_typeMistral.jsonl
#CUDA_VISIBLE_DEVICES=7 python eval/bon_eval_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --methods hinge --eval_mode min --dataset_file testset/math-llama3.1-8b-inst-64_with_rewards_typeMistral.jsonl

for model in "llama3-8b" "mistral-7b" ; do
    model_name=${model_name_map[${model}]}
    dataset_file=${dataset_file_map[${model}]}
    max_samples=${max_samples_map[${model}]}

    for method in "hinge" "dpo"; do
#      echo "Training PRM model $model_name with $method loss..."
#      CUDA_VISIBLE_DEVICES=1 python train/train_prm.py --model_name $model_name --max_samples $max_samples --methods $method --reward_batch_size 16
#      CUDA_VISIBLE_DEVICES=1 python eval/bon_eval.py --model_name $model_name --dataset_file $dataset_file --eval_mode min --methods $method
#      CUDA_VISIBLE_DEVICES=1 python eval/bon_eval.py --model_name $model_name --dataset_file $dataset_file --eval_mode orm --methods $method
      CUDA_VISIBLE_DEVICES=2 python eval/bon_eval.py --model_name $model_name --dataset_file $dataset_file --eval_mode min --methods $method
    done
done

