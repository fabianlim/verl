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

from typing import Callable, Optional, Sequence

try:
    # from math_verify.errors import TimeoutException
    from math_verify.parser import (
        ExprExtractionConfig, LatexExtractionConfig,
        NormalizationConfig,
        ExtractionTarget, parse
    )
    from math_verify.grader import verify

    import logging
    logger = logging.getLogger(__name__)
    # copied from  math-verify.metric.py

    def math_metric(
        gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
        pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
        precision: int = 6,
    ):

        def sample_level_fn(
            gold: str, pred: str,
        ) -> tuple[float, Optional[tuple[list[str], list[str]]]]:
            extracted_pred = parse(
                pred, pred_extraction_target,
                # raise_on_error=False, # DEBUG: make it not raise
            ) 
            extracted_gold = parse(
                "\\boxed{" + gold + "}", gold_extraction_target,
                # raise_on_error=False, # DEBUG: make it not raise
            )
            if len(extracted_pred) == 0:
                logger.warning(
                    f"We did not manage to extract a prediction in the correct format. Gold: {[gold]}, Pred: {[pred]}"
                )

            # We have to use timeout because the sypmy to str conversion can be very slow
            v = verify(
                extracted_gold, extracted_pred, precision, 
                # raise_on_error=False, # prevents verify from raising timeout
            )
            
            return (1.0 if v else 0.0), '[INVALID]'

        return sample_level_fn

    # https://github.com/volcengine/verl/issues/3407
    # NO TIMEOUT VERSION
    # def math_metric_no_raise(
    #     gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    #     pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    # ) -> Callable[
    #     [list[str], list[str]], tuple[float, Optional[tuple[list[str], list[str]]]]
    # ]:
    #     ret_score = verify(
    #         parse(gold_extraction_target[0], raise_on_error=False),
    #         parse(pred_extraction_target[0], raise_on_error=False),
    #         raise_on_error=False,
    #     )
    #     return ret_score, None

    # simpler version ported from skywork
    def math_metric_skywork(
        gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
        pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
        precision: int = 6,
    ):

        def sample_level_fn(ground_truth, solution_str):

            ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth

            pred = '[INVALID]'
            
            # 0 in case parsing cannot be completed
            try:
                math_verify_parsed = parse(
                    solution_str, 
                    pred_extraction_target,
                    parsing_timeout=5
                )
            except Exception:
                return 0.0, pred
            
            # 0 if parsing is problematic
            if len(math_verify_parsed) < 2:
                return 0.0, pred

            pred = math_verify_parsed[1]
            
            # We perform a quick string match first
            if pred in ground_truth:
                print ('quick')
                return 1.0, pred
            
            # We now fallback to semantic verification
            for gt in ground_truth:
                try:
                    if verify(
                        parse("\\boxed{" + gt + "}", gold_extraction_target, parsing_timeout=5),
                        math_verify_parsed,
                        timeout_seconds=5,
                    ):
                        return 1.0, pred
                except Exception:
                    continue
            
            # Very unlikely to be correct after the above matches
            return 0.0, pred
        return sample_level_fn

except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

def compute_score(
    model_output: str, ground_truth: str, timeout_score: float = 0, 
    search_last_chars: int = 300,
    use_skywork: bool = True,
    # use_timeout: bool = True,
) -> bool:

    normalization_config = NormalizationConfig(
        basic_latex=True,
        units=True,
        malformed_operators=False,
        nits=False,
        boxed="all",
        equations=False,
    )
    if use_skywork:
        METRIC_FUNC = math_metric_skywork
    else:
        METRIC_FUNC = math_metric

    verify_func = METRIC_FUNC(
        gold_extraction_target=(LatexExtractionConfig(
            normalization_config=normalization_config,
        ),),
        pred_extraction_target=(
            ExprExtractionConfig(), 
            LatexExtractionConfig(
                normalization_config=normalization_config,
            )
        ),
    )
    ret_score = 0.0

    # take the last part
    if search_last_chars is not None:
        model_output = model_output[-search_last_chars:]

    # Wrap the ground truth in \boxed{} format for verification
    preds = None
    try:
        ret_score, preds  = verify_func(ground_truth, model_output)
    except Exception:
        pass
    # except TimeoutException:
    except TimeoutError:
        ret_score = timeout_score

    return ret_score, preds
