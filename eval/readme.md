## Running Commands

### ImplicitPRM
To evaluate ImplicitPRM, you can run the following command:

```
python -m torch.distributed.launch --nproc_per_node=8 bon_eval.py \
         --load /home/openrlhf-checkpoints-final/ce-top8-2stage-8192 \
         --ref-load /home/models/Meta-Llama-3.1-8B-Instruct \
         --type implicit_prm
```

where `--load`, `--ref-load` indicates the path of your trained model and reference model. 

### Baseline: NTP-PRM

To evaluate the NTP-PRM baseline, you can run the following command:

```
python -m torch.distributed.launch --nproc_per_node=8 bon_eval.py \
        --load /home/openrlhf-checkpoints-final/sft-prm \
        --tokenizer-path /home/openrlhf-checkpoints-final/sft-prm \
        --type baseline-ntp \
        --begin-of-action-token <|reserved_special_token_0|> \
        --prm-token <|reserved_special_token_0|> \
```
