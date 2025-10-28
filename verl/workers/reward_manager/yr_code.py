# Copyright 2024 PRIME team and/or its affiliates
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

from collections import defaultdict

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from tqdm import tqdm
import numpy as np
from verl.utils.reward_score.skywork import compute_score as skywork_compute_score

def parallel_compute_score(
    evaluation_func, 
    response_str, 
    ground_truth, 
    timeout=6, 
    max_workers=64
):

    with tqdm(total=len(response_str)) as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    evaluation_func, 
                    response_str[index], 
                    ground_truth[index], 
                ): index
                for index in range(len(response_str))
            }
            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    return [results[i] for i in range(len(response_str))]

@register("yr")
class YRRewardManager:

    def __init__(
        self, tokenizer, num_examine, 
        compute_score=None, 
        overlong_buffer_cfg=None,
        max_resp_len=None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

        # OVERRIDE!!
        self.compute_score = skywork_compute_score # we always override, regardless what compute_score is
        
        # this is a different kind of overlong protection, where if a
        # sample hits the max response len, we will 
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [
            data_item.non_tensor_batch['reward_model']['ground_truth'] 
            for data_item in data
        ]
        ground_truth = [x.tolist() if isinstance(x, np.ndarray) else x for x in ground_truth]
        data_sources = data.non_tensor_batch['data_source']

        assert len(response_str) == len(ground_truth) == len(data_sources)

        scores = []
        batch_size = 1024
        try:
            for i in range(0, len(response_str), batch_size):
                cur_response_str = response_str[i:i+batch_size]
                cur_ground_truth = ground_truth[i:i+batch_size]

                cur_scores = parallel_compute_score(
                        self.compute_score,
                        cur_response_str,
                        cur_ground_truth,
                    )

                scores += cur_scores
            assert len(scores) == len(response_str)

        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0. for _ in range(len(response_str))]

        for i in range(len(data)):
            data_source = data_sources[i]
            reward = scores[i]
            if self.overlong_buffer_cfg.enable:
                overlong = self.max_resp_len <= valid_response_length[i].item()
                reward_extra_info['overlong'].append(overlong)
                reward_extra_info['overlong_response_len'].append(valid_response_length[i].item())

                if overlong:
                    # we know that scores are 1 for correct and 0 for wrong
                    # - so we just put a different value
                    reward = self.overlong_buffer_cfg.penalty_factor # HIJACK THIS

            if self.overlong_buffer_cfg.rep_tail:
                tail = self.overlong_buffer_cfg.rep_tail
                threshold = self.overlong_buffer_cfg.rep_threshold
                rep_spans = find_repetitions(
                    response_ids[i][
                        max(valid_response_length[i]-tail,0):
                        valid_response_length[i]
                    ].numpy(), 
                    return_spans=True
                )
                rep_tokens = sum([e-s for s,e in rep_spans])
                rep_ratio = rep_tokens / tail

                if rep_ratio > threshold:
                    reward = self.overlong_buffer_cfg.penalty_factor # HIJACK THIS

                reward_extra_info['overlong_rep_tokens'].append(rep_tokens)
                reward_extra_info['overlong_rep_ratio'].append(rep_ratio)
                reward_extra_info['overlong_rep_ratio_clip'].append(rep_ratio > threshold)
                reward_extra_info['overlong_rep_tail'].append(tail)
                reward_extra_info['overlong_rep_threshold'].append(threshold)


            reward_tensor[i, valid_response_length[i].item() - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[response]", response_str[i])


        reward_extra_info["acc"] = scores

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

# function for finding repetitions
def find_repetitions(
    tokens: List, 
    min_len: int = 2,
    return_spans: bool = False,
):

    # - map from position -> last seen index most
    #   recent
    most_recent_idx = {}

    # - store all the repeats in format
    #   (idx, start_of_repeat)
    repeats = []

    # - some constants
    N = len(tokens) # length

    # - 
    # - start of repeat
    # - current idx
    mri_prev, start_rep, n = None, -1, -1

    # loop over tokens
    while n < N:

        # advance 
        x = tokens[n]
        
        # index which the current token appeared before
        mri_curr = most_recent_idx.get(x)
        most_recent_idx[x] = n # update
        
        # - if x was seen before and one of two conds
        # 1. x was also the prev token (i.e, if tokens[n-1] == x)
        # 2. the prev token was seen one position behind position
        #    current token was also last seen
        if (
            (mri_curr is not None) and 
            (
                (n > 0 and x == tokens[n-1]) 
                or
                (mri_prev is not None and mri_curr == (mri_prev + 1))
            )
        ):
            repeats.append((n, start_rep))
        else:
            # this will be a new pattern, record
            # this pos as the start of (potential) future repeats
            start_rep = n 

        # for next loop
        mri_prev = mri_curr
        n += 1

    if not return_spans:
        return repeats

    # convert to spans
    spans = {}
    for e, s in repeats:
        spans[s] = max(e, spans.get(s,-1))
  
    return [
        (s,e+1) for s, e in spans.items() 
        if (e-s+1) >= min_len
    ]