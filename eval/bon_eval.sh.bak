MODEL_NAME="mistral-7b-bon"
CONFIG_NAME="config/7b"

CUR_DATE="notebook"

OPTS=""

# dpo version
OPTS+=" --load Windy0822/ImplicitPRM_DPO"
OPTS+=" --ref-load meta-llama/Llama-3.1-8B-Instruct"
OPTS+=" --type implicit_prm"

# prm as math-shepherd version
#OPTS+=" --load peiyi9979/math-shepherd-mistral-7b-prm"
#OPTS+=" --tokenizer-path peiyi9979/math-shepherd-mistral-7b-prm"
#OPTS+=" --type baseline-ntp"
#OPTS+=" --prm-token ки"

OPTS+=" --bon-dataset math" #choices: math gsm8k qa
OPTS+=" --batch-size 4"
OPTS+=" --baseline 0" # output pass@k and self-consistency@n if baseline=1
OPTS+=" --combine 0" # integrate self-consistency if combine=1

OPTS+=" $@"


CMD="python -m torch.distributed.launch --master_port 20550 --nproc_per_node=4 bon_eval.py ${OPTS}"
echo "${CMD}"
$CMD

