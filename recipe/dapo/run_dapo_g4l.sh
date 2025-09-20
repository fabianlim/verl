#!/usr/bin/env bash
set -xeuo pipefail

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
    +data.apply_chat_template_kwargs.thinking=${thinking:-True} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c:-10.0} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high:-0.28} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low:-0.2} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff:-0} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload:-True} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload:-True} \
    actor_rollout_ref.actor.grad_clip=${grad_clip:-1.0} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef:-0.0} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode:-"token-mean"} \
    actor_rollout_ref.actor.optim.lr=${lr:-1e-6} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps:-10} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay:-0.1} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu:-4} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.target_entropy=${target_entropy:-0.5} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size:-1} \
    actor_rollout_ref.actor.use_adaptive_entropy_adjustment=${use_adaptive_entropy_adjustment:-False} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz:-True} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss:-False} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload:-True} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz:-True} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size:-1} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz:-True} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs:-128} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt:-16} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${temperature:-1.0} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp:-1} \
    actor_rollout_ref.rollout.top_k="${top_k:--1}" \
    actor_rollout_ref.rollout.top_p=${top_p:-1.0} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_kwargs_n} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature:-1.0} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k:--1} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p:-0.7} \
    algorithm.adv_estimator=${adv_estimator:-grpo} \
    algorithm.filter_groups.enable=${enable_filter_groups:-True} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches:-10} \
    algorithm.filter_groups.metric=${filter_groups_metric:-acc} \
    algorithm.kl_ctrl.kl_coef=${kl_coef:-0.0} \
    algorithm.use_kl_in_reward=${use_kl_in_reward:-False} \
    data.gen_batch_size=${gen_batch_size} \
    data.max_prompt_length=${max_prompt_length:-2048} \
    data.max_response_length=${max_response_length:-8192} \
    data.prompt_key=prompt \
    data.train_batch_size=${train_batch_size:-156} \
    data.train_files="${TRAIN_FILE}" \
    data.truncation='left' \
    data.val_files="${TEST_FILE}" \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer:-False} \
    reward_model.overlong_buffer.len=${overlong_buffer_len:-4096} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor:-1.0} \
    reward_model.reward_manager=dapo \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.experiment_name="${exp_name}" \
    trainer.log_val_generations=${log_val_generations:-0} \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.project_name="${project_name}" \
    trainer.resume_mode=auto \
    trainer.rollout_data_dir=${rollout_data_dir:-null} \
    trainer.save_freq=${save_freq:-25} \
    trainer.test_freq=${test_freq:-5} \
    trainer.total_epochs=${total_epochs:-1} \
    trainer.validation_data_dir=${validation_data_dir:-null} \
    trainer.val_before_train=True
