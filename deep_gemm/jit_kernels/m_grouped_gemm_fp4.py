import torch
import warnings
from typing import Tuple, Optional

from .gemm_fp4 import get_best_configs, get_smem_config_fp4, check_mxfp4_scales_layout, preprocess_mxfp4_scales, _post_preprocess_mxfp4_scales
from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, GemmType, get_extra_info

# C++ code templates
includes = ('"../deep_gemm/fp4_gemm_cutlass3.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kEnableSboOverlap = {ENABLE_SBO_OVERLAP};
constexpr auto nExpand = {N_EXPAND};
constexpr auto hasBias = {HAS_BIAS};
constexpr auto kDynamicTileId = FP4DynamicTileId::{DynamicTileId};
constexpr auto kApplySwigluLimit = {kApplySwigluLimit};

// Make a templated grouped GEMM
using gemm_t = Fp4Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kEnableSboOverlap, hasBias, nExpand, kDynamicTileId, EpilogueType::{EPILOGUE_TYPE}, kApplySwigluLimit>;
gemm_t::run(lhs, lhs_scales, rhs, rhs_scales,
            bias, out, m, grouped_layout, block_m_info, expected_m,
            stream, num_sms, smem_size, signal, out_scale, swiglu_limit);
"""

def select_moe_dynamic_tile(n: int, k: int, expected_m: int, has_bias: bool) -> Tuple[bool, str]:
    """Decide whether to enable MoE dynamic-tile and which variant to use.

    Returns (enable_moe_dynamic_tile, dynamic_tile_id). dynamic_tile_id is a
    FP4DynamicTileId enum name string: 'LargeEM', 'LargeK', 'LargeK_G2', 'SmallEM';
    'Disabled' when not enabled.
    """
    extra_info = get_extra_info()
    env_use_moe_dynamic_tile = extra_info.get('use_moe_dynamic_tile', False)
    if has_bias or not env_use_moe_dynamic_tile:
        return False, 'Disabled'

    model_gemm1_shape_list = [
        (4096, 8192), # qwen3.8
    ]
    model_gemm2_shape_list = [
        (8192, 2048), # qwen3.8
    ]
    # gemm1
    if (n, k * 2) in model_gemm1_shape_list:
        #todo 修复大EM场景的stack问题
        if(expected_m < 6 or expected_m > 76):
            return False, 'Disabled'
    # gemm2
    elif (n, k * 2) in model_gemm2_shape_list:
        if(expected_m < 6 or expected_m > 51):
            return False, 'Disabled'
    else:
        if(expected_m < 23):
            return False, 'Disabled'
    # select dynamic-tile variant: LargeEM, LargeK, LargeK_G2, SmallEM
    if k > 2048:
        dynamic_tile_id = 'LargeEM' if expected_m > 32 else 'LargeK'
    else:
        if expected_m > 117:
            dynamic_tile_id = 'LargeEM'
        else:
            dynamic_tile_id = 'SmallEM' if k <= 1024 else 'LargeK_G2'
    return True, dynamic_tile_id

def m_grouped_gemm_fp4_fp4_bf16_nt_nopad(lhs_: Tuple[torch.Tensor, torch.Tensor],
                                         rhs_: Tuple[torch.Tensor, torch.Tensor],
                                         bias: torch.Tensor, out: torch.Tensor,
                                         m_indices: torch.Tensor, m_rows: torch.Tensor = None,
                                         configs = None) -> None:
    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    if (not check_mxfp4_scales_layout(scale=lhs_scales)):
        if not torch.compiler.is_compiling():
            warnings.warn("[DeepGemm] Called preprocess_mxfp4_scales for SFA inner GroupedNoPad interface.", UserWarning, stacklevel=3)
        lhs_scales = preprocess_mxfp4_scales(scale=lhs_scales)
    if (not check_mxfp4_scales_layout(scale=rhs_scales)):
        if not torch.compiler.is_compiling():
            warnings.warn(("[DeepGemm] Called preprocess_mxfp4_scales for SFB inner GroupedNoPad interface. "
                          "preprocess the weight scale might degrade the performance!"), UserWarning, stacklevel=3)
        ### forward compatibility for release_2v1
        rhs_scales = _post_preprocess_mxfp4_scales(scale=rhs_scales)
        if (not check_mxfp4_scales_layout(scale=rhs_scales)):
            rhs_scales = preprocess_mxfp4_scales(scale=rhs_scales)

    # Type and shape checks
    assert m == m_ == m__ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs.dtype == torch.uint8 and rhs.dtype == torch.uint8
    assert bias is None or bias.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()
    assert check_mxfp4_scales_layout(scale=lhs_scales) and check_mxfp4_scales_layout(scale=rhs_scales) and m_indices.is_contiguous()

    has_bias = True
    if bias is None: bias = torch.empty(0, dtype=torch.float32, device=lhs.device); has_bias = False

    # Do nothing if `m` is zero
    if m == 0:
        return

    expected_m = ceil_div(m, num_groups)

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()

    # TODO: enable fp4 get_best_configs
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, expected_m, n, k, num_groups, num_sms, gemm_type=GemmType.GroupedNoPad)
        # num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages = (num_sms, 256, 256, 128, 64, 64, 3)
        # smem_config = get_smem_config_fp4(num_stages, block_m, block_n, warp_m, warp_n, block_k)

    if m_rows is None:
        counts = torch.bincount(m_indices)
        min_n = min(counts.size(0), num_groups)
        experts_for_rows = torch.zeros(num_groups, dtype=torch.int32, device='cuda')
        if min_n > 0:
            experts_for_rows[:min_n] = counts[:min_n]
        m_rows = experts_for_rows

    ## the largest blockM_num is, num_groups - 1 only has 1 token, the last group has (m-1) tokens, blockM_num = num_group -1  + ceil_div(m + 1 - num_group, block_m)
    ## total line num: blockM_num + 1, line0 is used to store the real blockM_num
    ## total_size = (blockM_num + 1) * 4 * sizeof(int) Byte
    block_m_info = torch.empty((num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4, dtype=torch.int32, device=m_rows.device)
    n_expand = 1

    if k <= 512 and n % (block_n * 4) == 0 and not has_bias:
        n_expand = 4
    if k <= 128 and n % (block_n * 8) == 0 and not has_bias:
        n_expand = 8
    args = (lhs, lhs_scales, rhs, rhs_scales, bias, out, m, m_rows, block_m_info, expected_m, torch.cuda.current_stream(), num_sms, smem_config[0], torch.empty(0).int(), torch.empty(0, dtype=torch.uint16, device=out.device), 0.0)
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp4_fp4_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages, 'ENABLE_SBO_OVERLAP': False, 'GEMM_TYPE': 'GroupedNoPad', 'N_EXPAND' : n_expand, 'HAS_BIAS': has_bias, 'DynamicTileId': 'Disabled', 'EPILOGUE_TYPE': 'Default', 'kApplySwigluLimit': False},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.uint8), ('lhs_scales', torch.uint16),
                  ('rhs', torch.uint8), ('rhs_scales', torch.uint16),
                  ('bias', torch.float32), ('out', torch.bfloat16),
                  ('m', int), ('grouped_layout', torch.int32), ('block_m_info', torch.int32), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('signal', torch.int32), ('out_scale', torch.uint16), ('swiglu_limit', float)),
        template=template,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )

    # Run the kernel
    runtime(*args)

    return out

def m_grouped_gemm_fp4_fp4_bf16_nt_masked(lhs_: Tuple[torch.Tensor, torch.Tensor],
                                          rhs_: Tuple[torch.Tensor, torch.Tensor],
                                          bias: torch.Tensor, out: torch.Tensor,
                                          masked_m: torch.Tensor, expected_m: int, configs=None,
                                          enable_sbo_overlap: bool = False, signal: torch.Tensor = torch.empty(0).int(),
                                          out_scale: Optional[torch.Tensor] = None,
                                          swiglu_limit: Optional[float] = 0.0) -> None:
    """MoE GroupedMasked FP4 GEMM.

    When `out_scale` is not None, silu_and_mul + mxfp4 post-quant are fused into the epilogue:
    `out` must be uint8 of shape (num_groups, m, n // 4) and `out_scale` uint16 of shape
    (num_groups, m, ceil_div(n // 4, 32)). The gemm1 weight must have been interleaved with
    `preprocess_mxfp4_weight_for_act_and_quant_fusing` (before `preprocess_mxfp4_scales`), so
    that the epilogue reads gate/up (W1/W3) pairs from adjacent N positions.

    `swiglu_limit > 0` clamps before the activation: the gate is clamped from above only
    (`min(gate, limit)`) and the up projection on both sides (`clamp(up, -limit, limit)`);
    `0.0` or None disables the clamp.

    Only the masked layout is supported (nopad is not), `n` must be a multiple of 64, and a
    bias is not supported in the fused mode.

    NOTE: when `out_scale` is not None, it is re-strided **in place** on return to the N-major
    layout (sfm * sfn, 1, sfm) expected by the Gemm2 SFA reader.
    """

    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    num_groups, m, k = lhs.shape
    num_groups_, n, k_ = rhs.shape
    num_groups__, m_, n_ = out.shape

    enable_silu_and_mul_quant_fusing = out_scale is not None
    if enable_silu_and_mul_quant_fusing:
        shape_n_out = n // 4 ### /2: silu_and_mul; /2: quant mxfp4
        sfm, sfn = m, ceil_div(shape_n_out, 32)
        assert n_ == shape_n_out, f'{n_=}, expected {shape_n_out}'
        assert out_scale.shape == (num_groups, sfm, sfn), \
            f'out_scale shape {out_scale.shape}, expected {(num_groups, sfm, sfn)}'
        assert out.dtype == torch.uint8 and out_scale.dtype == torch.uint16
        assert bias is None, "bias not None is not supported in SiluAndMulPostQuant Epilogue."
        epilogue_type, output_type = 'SiluAndMulPostQuantFp4', torch.uint8
    else:
        assert n_ == n, f'{n_=}, expected {n}'
        out_scale = torch.empty(0, dtype=torch.uint16, device=out.device)
        assert out.dtype == torch.bfloat16
        epilogue_type, output_type = 'Default', torch.bfloat16
    num_groups___ = masked_m.numel()

    if (not check_mxfp4_scales_layout(scale=lhs_scales)):
        if not torch.compiler.is_compiling():
            warnings.warn("[DeepGemm] Called preprocess_mxfp4_scales for SFA inner GroupedMasked interface.", UserWarning, stacklevel=3)
        lhs_scales = preprocess_mxfp4_scales(scale=lhs_scales)
    if (not check_mxfp4_scales_layout(scale=rhs_scales)):
        if not torch.compiler.is_compiling():
            warnings.warn(("[DeepGemm] Called preprocess_mxfp4_scales for SFB inner GroupedMasked interface. "
                          "preprocess the weight scale might degrade the performance!"), UserWarning, stacklevel=3)
        ### forward compatibility for release_2v1
        rhs_scales = _post_preprocess_mxfp4_scales(scale=rhs_scales)
        if (not check_mxfp4_scales_layout(scale=rhs_scales)):
            rhs_scales = preprocess_mxfp4_scales(scale=rhs_scales)

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert lhs.dtype == torch.uint8 and rhs.dtype == torch.uint8
    assert bias is None or bias.dtype == torch.float32
    assert masked_m.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert check_mxfp4_scales_layout(scale=lhs_scales) and check_mxfp4_scales_layout(scale=rhs_scales)
    assert out.is_contiguous() and masked_m.is_contiguous()

    has_bias = True
    if bias is None: bias = torch.empty(0, dtype=torch.float32, device=lhs.device); has_bias = False

    if enable_sbo_overlap:
        assert signal is not None
        assert signal.is_contiguous()
        assert signal.dtype == torch.int32

    # Auto-tuning with compilation
    global includes, template

    num_sms = get_num_sms()
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, expected_m, n, k, num_groups, num_sms, gemm_type=GemmType.GroupedMasked)
        # num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages = (num_sms, 256, 256, 128, 64, 64, 3)
        # smem_config = get_smem_config_fp4(num_stages, block_m, block_n, warp_m, warp_n, block_k)
        if (enable_silu_and_mul_quant_fusing and block_n < 64):
            block_n = 64
    ## the largest blockM_num is, num_groups - 1 only has 1 token, the last group has (m-1) tokens, blockM_num = num_group -1  + ceil_div(m + 1 - num_group, block_m)
    ## total line num: blockM_num + 1, line0 is used to store the real blockM_num
    ## total_size = (blockM_num + 1) * 4 * sizeof(int) Byte
    block_m_info = torch.empty(
        (num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4, dtype=torch.int32, device=masked_m.device)

    n_expand = 1
    if k <= 512 and expected_m > 2 and n % (block_n * 4) == 0 and not has_bias and not enable_silu_and_mul_quant_fusing:
        n_expand = 4

    EnableMoeDynamicTile, DynamicTileId = select_moe_dynamic_tile(
        n, k, expected_m, has_bias)
    if EnableMoeDynamicTile:
        # fix the block config to avoid unecessary jit compile
        block_m, block_n, block_k, warp_m, warp_n, num_stages = (128, 128, 64, 64, 64, 3)

    swiglu_limit_ = 0.0 if (swiglu_limit is None) else swiglu_limit
    kApplySwigluLimit = swiglu_limit_ > 0

    args = (lhs, lhs_scales, rhs, rhs_scales, bias, out, m, masked_m, block_m_info, expected_m,
            torch.cuda.current_stream(), num_sms, smem_config[0], signal, out_scale, swiglu_limit_)
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp4_fp4_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages, 'ENABLE_SBO_OVERLAP': enable_sbo_overlap,
              'GEMM_TYPE': 'GroupedMasked', 'N_EXPAND' : n_expand, 'HAS_BIAS': has_bias,
              'DynamicTileId': DynamicTileId, 'EPILOGUE_TYPE': epilogue_type, 'kApplySwigluLimit': kApplySwigluLimit},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.uint8), ('lhs_scales', torch.uint16),
                  ('rhs', torch.uint8), ('rhs_scales', torch.uint16),
                  ('bias', torch.float32), ('out', output_type),
                  ('m', int), ('grouped_layout', torch.int32),
                  ('block_m_info', torch.int32), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('signal', torch.int32), ('out_scale', torch.uint16), ('swiglu_limit', float)),
        template=template,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )

    # Run the kernel
    runtime(*args)

    if enable_silu_and_mul_quant_fusing:
        ### sfm and sfn always > 1 for GroupedMasked(topk > 1).
        out_scale.as_strided_(size=(num_groups, sfm, sfn), stride=(sfm * sfn, 1, sfm))

    return (block_m, ceil_div(n, block_n))
