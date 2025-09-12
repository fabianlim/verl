#!/usr/bin/env bash
set -xeuo pipefail

# NOTE: @goon - flim is skeptical of mamba tp, might need to turn it off for quality
# TODO: @goon - follow /proj/data-eng/goon/verl/recipe/skywork/run_skywork_7b_16k.sh

actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
adv_estimator=${adv_estimator:-grpo}
clip_ratio_high=${clip_ratio_high:-0.28}
clip_ratio_low=${clip_ratio_low:-0.2}
enable_filter_groups=${enable_filter_groups:-True}
enable_overlong_buffer=${enable_overlong_buffer:-False}
entropy_coeff=${entropy_coeff:-0}
exp_name=${exp_name:-'g4l-test'}
filter_groups_metric=${filter_groups_metric:-acc}
gen_prompt_bsz=${gen_prompt_bsz:-$((train_prompt_bsz))}
gen_tp=${gen_tp:-1}
grad_clip=${grad_clip:-1.0}
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
kl_coef=${kl_coef:-0.0}
kl_loss_coef=${kl_loss_coef:-0.0}
loss_agg_mode=${loss_agg_mode:-"token-mean"}
lr=${lr:-1e-6}
lr_warmup_steps=${lr_warmup_steps:-10}
max_num_gen_batches=${max_num_gen_batches:-10}
max_prompt_length=${max_prompt_length:-$((1024 * 2))}
max_response_length=${max_response_length:-$((1024 * 8))}
n_resp_per_prompt=${n_resp_per_prompt:-16}
offload=${offload:-True}
overlong_buffer_len=${overlong_buffer_len:-$((1024 * 4))}
overlong_penalty_factor=${overlong_penalty_factor:-1.0}
project_name=${project_name:-'DAPO'}
save_freq=${save_freq:-50}
sp_size=${sp_size:-1}
temperature=${temperature:-1.0}
test_freq=${test_freq:-5}
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
top_p=${top_p:-1.0}
total_epochs=${total_epochs:-1}
train_prompt_bsz=${train_prompt_bsz:-256}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-32}
use_dynamic_bsz=${use_dynamic_bsz:-True}
use_kl_in_reward=${use_kl_in_reward:-False}
use_kl_loss=${use_kl_loss:-False}
val_top_p=${val_top_p:-0.7}
weight_decay=${weight_decay:-0.1}


# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-16}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is not set"
    exit 1
fi
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# TODO: @goon - entropy cfgs?
# actor_rollout_ref.actor.use_adaptive_entropy_adjustment=$USE_ADAPTIVE_ENT \
# actor_rollout_ref.actor.target_entropy=${TGT_ENTROPY} \

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --submission-id g4l \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
