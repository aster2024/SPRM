#!/bin/bash

model_name=meta-llama/Llama-3.1-8B-Instruct
dataset_file=testset/math-llama3.1-8b-inst-64.json
CUDA_VISIBLE_DEVICES=7 python eval/bon_eval.py --model_name $model_name --dataset_file $dataset_file --reward_model_load model/reward_model_svm_meta-llama_Llama-3.1-8B-Instruct.pt --model_type svm --max_samples 500
CUDA_VISIBLE_DEVICES=7 python eval/bon_eval.py --model_name $model_name --dataset_file $dataset_file --reward_model_load model/reward_model_linear_meta-llama_Llama-3.1-8B-Instruct.pt --model_type linear --max_samples 500