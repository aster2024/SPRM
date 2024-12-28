#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:`realpath .`

beta=0.05
train_dir=ce-data
exp_name=ce
save_steps=-1
max_epochs=1
LR=5e-7


datapath=./processed_data/$train_dir
modelpath=meta-llama/Llama-3.1-8B-Instruct

save_steps=$save_steps
max_epochs=$max_epochs
exp_name=$exp_name

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ce \
   --save_path ./openrlhf-checkpoints-final/$exp_name \
   --ckpt_path ./openrlhf-checkpoints-final/$exp_name \
   --max_ckpt_num 50 \
   --save_steps $save_steps \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 64 \
   --micro_train_batch_size 2 \
   --pretrain $modelpath \
   --bf16 \
   --max_epochs $max_epochs \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate $LR \
   --beta $beta \
   --dataset $datapath \
   --apply_chat_template \
   --chosen_key response \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing
EOF


deepspeed --module $training_commands