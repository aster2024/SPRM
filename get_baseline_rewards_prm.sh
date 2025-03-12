#!/bin/bash

#echo "Getting PRM rewards for training data using Mistral model..."
#CUDA_VISIBLE_DEVICES=3 python train/get_baseline_rewards.py --reward_name_or_path RLHFlow/Llama3.1-8B-PRM-Mistral-Data --input_file data/ultrainteract_math_rollout.json --model_type Mistral
#echo "Getting PRM rewards for training data using Deepseek model..."
#CUDA_VISIBLE_DEVICES=3 python train/get_baseline_rewards.py --reward_name_or_path RLHFlow/Llama3.1-8B-PRM-Deepseek-Data --input_file data/ultrainteract_math_rollout.json --model_type Deepseek

for input_file in "testset/math-llama3.1-8b-inst-64.json" "testset/math-Mistral-7B-Instruct-v0.2-64.json" "testset/math-Meta-Llama-3.1-70B-Instruct-64.json" ; do
#  echo "Getting PRM rewards for $input_file using Mistral models..."
#  CUDA_VISIBLE_DEVICES=3 python eval/get_baseline_rewards.py --reward_name_or_path RLHFlow/Llama3.1-8B-PRM-Mistral-Data --input_file $input_file --model_type Mistral
#  echo "Getting PRM rewards for $input_file using Deepseek models..."
#  CUDA_VISIBLE_DEVICES=3 python eval/get_baseline_rewards.py --reward_name_or_path RLHFlow/Llama3.1-8B-PRM-Deepseek-Data --input_file $input_file --model_type Deepseek
done
