#!/bin/bash
set -ex

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

# Entropy Config
ENTROPY_COEFF=0.0
USE_ADAPTIVE_ENT=True
TGT_ENTROPY=0.2
# MAX_ENT_COEF=0.005
# MIN_ENT_COEF=0
DELTA_ENT_COEF=0.0001

ROLLOUT_BATCH_SIZE=256
PPO_MINI_BATCH=256
MAX_PROMPT_LENGTH=2048
RES_LENGTH=8192
GROUP_SIZE=1
N_VAL_SAMPLES=8 # REMOVED

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

TRAIN_TEMPERATURE=1.0

TP=1
SP=1
MAX_TOKEN_LEN=$(((RES_LENGTH + MAX_PROMPT_LENGTH + 1000) / SP))

# Your Model Path
MODEL_PATH=${MODEL_PATH:-}
CODE_PATH=${CODE_PATH:-}
if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is not set"
    exit 1
fi
if [ -z "$CODE_PATH" ]; then
    echo "CODE_PATH is not set"
    exit 1
fi

# Since math queries are much more than code queries, we duplicate the math data when mixing the datasets
#train_files="[\"$CODE_PATH/or1_data/train/train_7b_code.pkl\",\"$CODE_PATH/or1_data/train/train_7b_code.pkl\",\"$CODE_PATH/or1_data/train/train_7b_math.pkl\"]"
train_files="[\"$CODE_PATH/or1_data/train/train_7b_math.pkl\"]"
test_files="[\"$CODE_PATH/or1_data/eval/aime24.parquet\",\"$CODE_PATH/or1_data/eval/aime25.parquet\"]"

PROJECT_NAME=skywork-or1-train

EXP_NAME=7B_L$(($RES_LENGTH / 1024))k
MODEL_NAME=$(basename $MODEL_PATH)
EXP_NAME=$EXP_NAME-${MODEL_NAME}-bs${ROLLOUT_BATCH_SIZE}-minibs${ROLLOUT_BATCH_SIZE}-gs${GROUP_SIZE}-tgt${TGT_ENTROPY}-temp${TRAIN_TEMPERATURE}-${WORLD_SIZE}nodes
SAVE_DIR=$CODE_PATH/verl_ckpt/$PROJECT_NAME/$EXP_NAME
SAVE_STATS_DIR=${SAVE_DIR}/stats
mkdir -p $SAVE_DIR
mkdir -p $SAVE_STATS_DIR


python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=$ROLLOUT_BATCH_SIZE \
    data.val_batch_size=13000 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$RES_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.use_adaptive_entropy_adjustment=$USE_ADAPTIVE_ENT \
    actor_rollout_ref.actor.target_entropy=${TGT_ENTROPY} \
    actor_rollout_ref.actor.entropy_coeff_delta=${DELTA_ENT_COEF} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=20 \
    trainer.test_freq=20\
    trainer.default_local_dir=$SAVE_DIR \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}"
    