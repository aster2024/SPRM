## Running Commands

### ImplicitPRM
- To evaluate ImplicitPRM using **BoN**, you can run the following command:

```
python -m torch.distributed.launch --nproc_per_node=8 bon_eval.py \
         --load /home/openrlhf-checkpoints-final/ce-top8-2stage-8192 \
         --ref-load /home/models/Meta-Llama-3.1-8B-Instruct \
         --type implicit_prm
```

where `--load`, `--ref-load` indicates the path of your trained model and reference model. 

- To evaluate ImplicitPRM using **ProcessBench**, you can run the following command for inference and evaluation.

**Inference:**
```
python processbench.py \
    --mode inference \
    --input_file /path/to/processbench \
    --output_file /path/to/output \
    --model_path /path/to/EurusPRM \
    --ref_model_path /path/to/ref_model \
    --tokenizer_path /path/to/EurusPRM \
    --batch_size 2 \
    --coef 0.001
```

**Evaluation:**

```
python processbench.py \
    --mode evaluate \
    --input_file /path/to/pred \
    --output_file /path/to/output \
    --num_thresholds 2000
```

where `--num_thresholds` indicates the number of slices that will be used for the selection of the optimal threshold.

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
