#!/bin/bash

#echo "Getting ORM rewards for training data by Armo"
#CUDA_VISIBLE_DEVICES=0 python train/get_baseline_rewards_orm.py --reward_name_or_path RLHFlow/ArmoRM-Llama3-8B-v0.1 --input_file data/ultrainteract_math_rollout.json --model_type Armo

for input_file in "testset/math-llama3.1-8b-inst-64.json" "testset/math-Mistral-7B-Instruct-v0.2-64.json" "testset/math-Meta-Llama-3.1-70B-Instruct-64.json" ; do
  echo "Getting ORM rewards for $input_file by Armo"
  CUDA_VISIBLE_DEVICES=0 python eval/get_baseline_rewards_orm.py --reward_name_or_path RLHFlow/ArmoRM-Llama3-8B-v0.1 --input_file $input_file --model_type Armo
  echo "Getting ORM rewards for $input_file by Skywork"
  CUDA_VISIBLE_DEVICES=0 python eval/get_baseline_rewards_orm.py --reward_name_or_path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 --input_file $input_file --model_type Skywork
done
