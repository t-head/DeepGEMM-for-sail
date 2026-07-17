import json, os
import pytest
from deep_gemm.deep_gemm_tuner import (
    autotune_deepgemm,
    benchmark_deepgemm
)

seed = 123
performance_threshold = 0.85
test_cases_dir = os.path.dirname(os.path.abspath(__file__))+"/tune_configs/"
test_cases_path = {
    "dense": test_cases_dir+"dense_configs.json",
    "masked": test_cases_dir+"masked_configs.json",
    "nopad": test_cases_dir+"nopad_configs.json",
    "nopad_bf16": test_cases_dir+"nopad_bf16_configs.json",
    "masked_bf16": test_cases_dir+"masked_bf16_configs.json",
}

def gen_test_case(mnk_configs: str):
    # load mnk file
    with open(mnk_configs, "r") as f:
        test_cases = json.load(f)
        return test_cases

@pytest.mark.parametrize("test_configs", ["dense", "masked", "nopad", "nopad_bf16"])
def test_deepgemm_tuning(test_configs):
    os.environ["DEEPGEMM_TUNER_DEBUG_MODE"] = "1"
    save_path = autotune_deepgemm.tuning_deepgemm_config_entrypoint(
        gen_test_case(test_cases_path[test_configs]),
        1,
        seed,
        None,
        out_of_box=False,
    )
    os.environ["DEEPGEMM_TUNER_DEBUG_MODE"] = "0"
    # check if best_configs is tuned
    assert os.path.exists(save_path), f"Tuned result not found at {save_path}"
    benchmark_result = benchmark_deepgemm.benchmark_increment(save_path)
    for result in benchmark_result:
        assert result[-1] > performance_threshold, f"{result[:-1]} gets poorer perfomance than baseline ({result[-1]}<={performance_threshold})"
    # os.remove(save_path)
 

if __name__ == '__main__':
    pytest.main([__file__, "-v"])
