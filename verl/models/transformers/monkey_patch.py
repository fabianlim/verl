# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Apply monkey-patch function to models
"""

import importlib.metadata
import sys
from functools import lru_cache
from typing import Optional, Callable

import torch
from packaging import version
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel

from verl.utils.import_utils import is_trl_available
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)

import os
PATCH_SDPA_SETTING = os.environ.get('PATCH_SDPA', 'JAGGED')

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    flash_impl: Callable = _flash_attention_forward,
    flash_impl_head_bef_seq: bool = True,
    support_gqa: bool = True,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()


    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:

        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"

        if flash_impl_head_bef_seq:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
        # For example:
        # - nheads_k=4, sp=8, repeats=2
        # - nheads_k=8, sp=8, repeats=1
        # - nheads_k=16, sp=8, repeats=1
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate
        # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.
        # https://github.com/huggingface/transformers/pull/33932

        # (bsz, seq_len/n) -> (bsz, seq_len)
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)

        if not support_gqa:
            repeats = query_states.size(2) // key_states.size(2)
            key_states = repeat_kv(key_states, repeats)
            value_states = repeat_kv(value_states, repeats)

        if flash_impl_head_bef_seq:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
    else:
        if not support_gqa:
            # transpose here really means hdim == 1
            assert flash_impl_head_bef_seq
            repeats = query_states.size(1) // key_states.size(1)
            key_states = torch.repeat_interleave(key_states, repeats, dim=1)
            value_states = torch.repeat_interleave(value_states, repeats, dim=1)

    attn_output = flash_impl(
        query_states, key_states, value_states, *args, position_ids=position_ids, **kwargs
    )
    # output: (bsz, seq_len, n_head/n, head_dim)

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


def patch_vlm_for_ulysses_input_slicing(model_class: type):
    """
    Applies a monkey patch to the forward method of a given model class
    to enable Ulysses sequence parallelism input slicing.
    """

    def _create_ulysses_wrapped_decoder_forward(original_forward):
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

            slice_now = (
                inputs_embeds is not None
                and current_ulysses_sp_size > 1
                and getattr(self, "_needs_initial_slice", True)
            )
            if slice_now:
                call_kwargs["inputs_embeds"] = slice_input_tensor(inputs_embeds, dim=1, padding=False)
                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    original_forward = model_class.forward
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward
    print(f"Monkey patch {model_class.__name__}.forward for Ulysses SP input slicing.")


def patch_forward_with_backends(
    model: PreTrainedModel,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
):
    """
    Choose the forward function based on the model and backend.
    Args:
        model (PreTrainedModel): The model to apply the monkey patch.
        use_fused_kernels (bool): Whether to use fused kernels.
        fused_kernels_backend (str): The backend to use for fused kernels.
    """
    if not use_fused_kernels or fused_kernels_backend not in ["triton", "torch"]:
        print(
            f"Skipping monkey patch for {model.__class__.__name__} as use_fused_kernels is "
            f"{use_fused_kernels} or fused_kernels_backend is {fused_kernels_backend}"
        )
        return

    forward_with_torch_backend_function = model.__class__.forward
    forward_with_triton_backend_function = model.__class__.forward
    if model.config.model_type == "qwen2_5_vl":
        from verl.models.transformers.qwen2_5_vl import forward_with_torch_backend, forward_with_triton_backend

        forward_with_torch_backend_function = forward_with_torch_backend
        forward_with_triton_backend_function = forward_with_triton_backend
    elif model.config.model_type == "qwen2_vl":
        from verl.models.transformers.qwen2_vl import forward_with_torch_backend, forward_with_triton_backend

        forward_with_torch_backend_function = forward_with_torch_backend
        forward_with_triton_backend_function = forward_with_triton_backend
    else:
        from verl.models.transformers.dense_common import forward_with_torch_backend, forward_with_triton_backend

        forward_with_torch_backend_function = forward_with_torch_backend
        forward_with_triton_backend_function = forward_with_triton_backend

    if fused_kernels_backend == "triton":
        model.__class__.forward = forward_with_triton_backend_function
        print(f"Using Triton backend for fused kernels in {model.__class__.__name__}")
    elif fused_kernels_backend == "torch":
        model.__class__.forward = forward_with_torch_backend_function
        print(f"Using Torch backend for fused kernels in {model.__class__.__name__}")
    else:
        raise ValueError(f"Unsupported fused_kernels_backend: {fused_kernels_backend}. Choose 'triton' or 'torch'.")


def apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
    use_remove_padding: bool = True,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
):
    """
    Apply monkey patch to the models for ulysses sequence parallel and fused kernel.

    In the end of this function forward function of the model is patched for fused kernel.
    If the model is not supported with fused kernel, please return after patch.
    """

    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    module = sys.modules[model.__module__]

    try:
        num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    assert num_attention_heads % ulysses_sp_size == 0, (
        f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
    )
    assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0, (
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size "
        f"{ulysses_sp_size}or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,"
        f"kv heads are repeated to ensure correctness."
    )

    if is_trl_available():
        from trl import AutoModelForCausalLMWithValueHead  # type: ignore

        def state_dict(self, *args, **kwargs):
            return torch.nn.Module.state_dict(self, *args, **kwargs)

        AutoModelForCausalLMWithValueHead.state_dict = state_dict
        print("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")

    # TODO: VLM models only, unify monkey patch to LLM models.
    if model.config.model_type == "qwen2_5_vl":
        if is_transformers_version_in_range(min_version="4.53.0"):
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLFlashAttention2 as Qwen2_5_VLAttention,
            )

        if use_remove_padding or ulysses_sp_size > 1:
            from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward

            Qwen2_5_VLAttention.forward = ulysses_flash_attn_forward
            print("Monkey patch FlashAttention2.forward in Qwen2.5VL")

        if ulysses_sp_size > 1:
            if is_transformers_version_in_range(min_version="4.52.0"):
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

                patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)
            else:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

                patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLModel)

    elif model.config.model_type == "qwen2_vl":
        if is_transformers_version_in_range(min_version="4.53.0"):
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention
        else:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2 as Qwen2VLAttention

        if use_remove_padding or ulysses_sp_size > 1:
            from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward

            Qwen2VLAttention.forward = ulysses_flash_attn_forward
            print("Monkey patch FlashAttention2.forward in Qwen2VL")

        if ulysses_sp_size > 1:
            if is_transformers_version_in_range(min_version="4.52.0"):
                from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLTextModel

                patch_vlm_for_ulysses_input_slicing(Qwen2VLTextModel)
            else:
                from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

                patch_vlm_for_ulysses_input_slicing(Qwen2VLModel)

    elif model.config.model_type == "kimi_vl":
        if use_remove_padding or ulysses_sp_size > 1:
            # TODO: Changes need to be made when transformers are adapted.
            from verl.models.transformers.kimi_vl import _ulysses_flash_attn_forward

            module.DeepseekV3FlashAttention2.forward = _ulysses_flash_attn_forward
            print("Monkey patch FlashAttention2.forward in KimiVL")

        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(module.DeepseekV3ForCausalLM)

        if use_fused_kernels:
            print("Not support fused kernels for KimiVL")

        return

    # transformers<=4.47.1
    if use_remove_padding or ulysses_sp_size > 1:
        if hasattr(module, "_flash_attention_forward"):
            module._flash_attention_forward = _ulysses_flash_attention_forward
            print(f"Monkey patch _flash_attention_forward in {model.__module__}")
        else:
            # transformers>=4.48.0
            from transformers.integrations import flash_attention

            flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
            print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")

            patch_spda, support_gqa, jagged = {
                'DISABLED': (False, None, None),
                'MASK': (True, True, False),
                'JAGGED': (True, False, True),
                'JAGGED_FLEX': (True, True, "flex"),
            }[PATCH_SDPA_SETTING]

            if not patch_spda:
                print(
                    "Disabled Monkey patch sdpa_attention_forward. "
                )
                assert ulysses_sp_size == 1, "patching of sdpa_attention_forward required for SP"
            else:
                from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
                def _sdpa_attention_forward(
                    module: torch.nn.Module,
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    attention_mask: Optional[torch.Tensor],
                    dropout: float = 0.0,
                    scaling: Optional[float] = None,
                    is_causal: Optional[bool] = True,
                    **kwargs,
                ) -> tuple[torch.Tensor, None]:

                    def _flash_impl(
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        atten: torch.Tensor,
                        *args,
                        position_ids=None,
                        **kwargs,
                    ):
                        if not jagged:
                            rebuild_attn = False
                            if ulysses_sp_size > 1:
                                rebuild_attn = True
                                assert position_ids is not None
                                position_ids = position_ids.squeeze(0)
                                blocks, = torch.where(position_ids == 0)
                                blocks = blocks.tolist() + [position_ids.shape[-1]]
                                chunks = torch.zeros(
                                    [position_ids.shape[-1]], dtype=torch.int, 
                                    device=position_ids.device
                                )
                                if len(blocks) > 2:
                                    for i in range(1, len(blocks) -1 ):
                                        chunks[blocks[i]:blocks[i+1]] = i

                                atten = (position_ids[:, None] >= position_ids) * (chunks[:, None] == chunks)
                                atten = atten[None, None:, :]

                            attn_output = torch.nn.functional.scaled_dot_product_attention(
                                q,
                                k,
                                v,
                                attn_mask=atten,
                                dropout_p=dropout,
                                scale=scaling,
                                is_causal=atten is None,
                                enable_gqa=True
                            ) # b, h, s, d

                            attn_output = attn_output.transpose(1,2).contiguous()
                            if rebuild_attn:
                                del atten
                            return attn_output

                        from transformers.modeling_flash_attention_utils import  prepare_fa_kwargs_from_position_ids
                        (sizes, _), __ = prepare_fa_kwargs_from_position_ids(position_ids)
                        sizes = sizes.diff().tolist()
                        if len(sizes) > 1:
                            q = torch.nested.as_nested_tensor(
                                torch.split(q.squeeze(0), sizes, dim=1), 
                                layout=torch.jagged,
                            )
                            k = torch.nested.as_nested_tensor(
                                torch.split(k.squeeze(0), sizes, dim=1), 
                                layout=torch.jagged,
                            )
                            v = torch.nested.as_nested_tensor(
                                torch.split(v.squeeze(0), sizes, dim=1), 
                                layout=torch.jagged,
                            )
                        if jagged == 'flex':
                            def causal_mask(score, b, h, q_idx, kv_idx):
                                return torch.where(q_idx >= kv_idx, score, -float("inf"))
                            attn_output = torch.nn.attention.flex_attention.flex_attention(
                                q,
                                k,
                                v,
                                score_mod=causal_mask,
                                scale=scaling,
                                enable_gqa=True,
                            ) # b, h, s, d
                        else:
                            attn_output = torch.nn.functional.scaled_dot_product_attention(
                                q,
                                k,
                                v,
                                attn_mask=None,
                                dropout_p=dropout,
                                scale=scaling,
                                is_causal=True,
                            ) # b, h, s, d
                        if len(sizes) > 1:
                            attn_output = torch.concat(
                                list(attn_output), dim=1
                            ).unsqueeze(0)

                        attn_output = attn_output.transpose(1,2).contiguous()
                        return attn_output

                    out = _ulysses_flash_attention_forward(
                        query, key, value, 
                        attention_mask,
                        flash_impl=_flash_impl,
                        dropout=dropout,
                        scaling=scaling,
                        is_causal=is_causal,
                        flash_impl_head_bef_seq=True, # kernel takes h,s
                        support_gqa=support_gqa,
                        **kwargs,
                    )
                    return out, None

                ALL_ATTENTION_FUNCTIONS['sdpa'] = _sdpa_attention_forward
                print(f"Monkey patch sdpa_attention_forward in with jagged={jagged}")

    patch_forward_with_backends(model, use_fused_kernels=use_fused_kernels, fused_kernels_backend=fused_kernels_backend)


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check
