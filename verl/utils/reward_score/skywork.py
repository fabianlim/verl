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

# Copied from https://github.com/SkyworkAI/Skywork-OR1

import json
import re
import traceback
from math_verify import parse, verify
import tempfile
import subprocess
from contextlib import contextmanager
import signal
import ast
import numpy as np
from typing import Dict

try:
    # requires an installation of livecodebench
    # - unfortunately its not packaged properly, so do an install like this
    # git clone https://github.com/LiveCodeBench/LiveCodeBench.git 
    # cd LiveCodeBench
    # pip install -e .
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
except ImportError:
    pass

# helper function
@contextmanager
def timeout_run(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError(
            f"Coderun timed-out after {seconds} seconds"
        )
    
    # 注册信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

# helper function
def convert_function_to_class_method(
    raw_code: str, 
    function_name: str
) -> str:

    # parse the code tree
    tree = ast.parse(raw_code)
    target_func = None
    new_body = []

    # Traverse the top-level nodes and keep the code of non-target functions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target_func = node
        else:
            new_body.append(node)
    
    if target_func is None:
        return None

    # - TODO: what is this??
    if (
        not (
            target_func.args.args and 
            target_func.args.args[0].arg == "self"
        )
    ):
        self_arg = ast.arg(arg="self", annotation=None)
        target_func.args.args.insert(0, self_arg)    

    # convert to a class definition
    class_def = ast.ClassDef(
        name="Solution",
        bases=[],
        keywords=[],
        body=[target_func],
        decorator_list=[]
    )
    
    new_body.append(class_def)
    tree.body = new_body
    
    # unpase the code tree
    new_code = ast.unparse(tree)
    return new_code

# for verifying math 
def math_verify_reward_function(solution_str, ground_truth):

    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return 0.0

# compute score function
def compute_score(
    completion: str, 
    ground_truth: Dict, 
    timeout: int = 6, 
):
    # seperate the solution from think tags
    if "</think>" in completion:
        solution_str = completion.split("</think>")[1]
    else:
        solution_str = completion

    # clear out all the None
    ground_truth = {
        k:v for k, v in ground_truth.items() 
        if v is not None
    }

    try:
        # if its math
        if 'math_verify' in ground_truth:
            return (
                math_verify_reward_function(
                    solution_str, 
                    ground_truth['math_verify']
                ), 
                ('math-verify', )
            )

        # otherwise it is pythonic code, so look for it
        solutions = re.findall(
            r"```python\n(.*?)```", 
            solution_str, re.DOTALL
        )
        if len(solutions) == 0:
            return False, ("code-no-solutions",)

        # syntax check the pythonic code
        # - if more than one, take the last instance
        solution = solutions[-1]
        try:
            ast.parse(solution)
        except:
            # if fails to parse return
            return False, ("code-fail-ast",)

        # CASE I: Test by writing to a file
        if (
            'import_prefix' in ground_truth and
            'test_code' in ground_truth and
            'entry_point' in ground_truth
        ):
            # get solution and test code
            solution = ground_truth["import_prefix"] + solution
            test_code = [
                x for x in ground_truth['test_code'].split("\n") if len(x) > 0
            ]

            unit_test_result = []
            unit_test_metadata = []
            for i in range(1, len(test_code)):
                cur_solution = solution
                cur_solution += "\n" + test_code[0] + test_code[i]
                cur_solution += "\ncheck({})".format(
                    ground_truth['entry_point']
                )

                try:
                    # Execute
                    with timeout_run(seconds=2):
                        with tempfile.NamedTemporaryFile(
                            mode='w', suffix='.py'
                        ) as temp_file:
                            temp_file.write(cur_solution)
                            temp_file.flush()
                            result = subprocess.run(
                                ['python', temp_file.name],
                                capture_output=True,
                                text=True,
                                timeout=timeout
                            )
                            if result.returncode != 0:
                                unit_test_result.append(False)
                                unit_test_metadata.append(
                                    f"Error: {result.stderr}"
                                )
                            else:
                                unit_test_result.append(True)
                                unit_test_metadata.append(f"Success")
                except TimeoutError as e:
                    unit_test_result.append(False)
                    unit_test_metadata.append(str(e))
                except Exception as e:
                    unit_test_result.append(False)
                    unit_test_metadata.append(
                        f"Execution exception: {str(e)}"
                    )
                    
            # binarize and return
            return all(unit_test_result), (
                'code-1',
                unit_test_metadata
            )

        # CASE 2: Test using livecodebench
        if (
            'inputs' in ground_truth and
            'outputs' in ground_truth 
        ):
            assert isinstance(ground_truth, dict)

            if (
                "fn_name" in ground_truth and 
                "class Solution" not in solution
            ):
                # need to format a solution class
                solution = convert_function_to_class_method(
                    solution, ground_truth["fn_name"]
                )
                if not isinstance(solution, str):
                    return False, None
            
            # call the livecodebench check_correctness function
            metrics = check_correctness(
                {
                    "input_output": json.dumps(ground_truth)
                },
                solution,
                debug=False,
                timeout=timeout,
            )

            metrics = list(metrics)
            fixed = []
            for e in metrics[0]:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            metrics[0] = fixed

            # binarize and return
            return sum(metrics[0]) == len(metrics[0]), (
                'code-lcb',
                metrics
            )

        # unknown case
        raise RuntimeError("Cannot handle example")

    except:
        # fallback
        traceback.print_exc(10)
        return False, None

    