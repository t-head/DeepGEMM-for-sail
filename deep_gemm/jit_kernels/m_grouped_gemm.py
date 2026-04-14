import torch
from typing import Tuple

from .gemm import get_best_configs, get_gemv_best_configs
from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_extra_info, is_ppu1v5_device
import os

# C++ code templates
includes = ('"deep_gemm/bf16_gemm.cuh"', )
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

// Make a templated grouped GEMM
using gemm_t = Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kEnableSboOverlap>;

// Launch kernel
gemm_t::run(out, grouped_layout, block_m_info,
            m, expected_m, lhs, rhs,
            stream, num_sms, smem_size, signal);
"""

includes_cutlass3 = ('"../deep_gemm/bf16_gemm_cutlass3.cuh"', )
template_cutlass3 = """
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

// Make a templated grouped GEMM
using gemm_t = Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kEnableSboOverlap, KernelType::{KERNEL_TYPE}>;

// Launch kernel
gemm_t::run(out, grouped_layout, block_m_info,
            m, expected_m, lhs, rhs,
            stream, num_sms, smem_size, signal);
"""

includes_gemv = ('"deep_gemm/gemvt.cuh"', )
template_gemv = """
using namespace deep_gemm;

// Templated args from Python JIT call
using D = __nv_bfloat16;
using acc_D = {acc_type};
constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto ThreadPerN = {ThreadPerN};
constexpr auto NPerThread = {NPerThread};
constexpr auto NUM_UNROLL = {NUM_UNROLL};
constexpr auto SWZL_SIZE_M = {SWZL_SIZE_M};
constexpr auto USE_SMALL_K = {USE_SMALL_K};
constexpr auto BlockSize = {BlockSize};

// Make a templated grouped GEMM
using gemm_v = Gemvt<D, D, acc_D, N, K, kNumGroups, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M, BlockSize, USE_SMALL_K>;

// Launch kernel
gemm_v::run(out, grouped_layout,
            m, lhs, rhs,
            stream);
"""

includes_fusedmoe_gemm = ('"../deep_gemm/fused_moe_gemm.cuh"', )
template_fusedmoe_gemm = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kEnableSboOverlap = {ENABLE_SBO_OVERLAP};

// Make a templated grouped GEMM
using fused_moe_gemm = FusedMoeGemm<N, K, kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumStages,
                                    GemmType::{GEMM_TYPE}, kEnableSboOverlap, KernelType::{KERNEL_TYPE}>;

// Launch kernel
fused_moe_gemm::run(out, grouped_layout, block_m_info,
            sorted_token_ids, expert_ids, block_m_offset,
            m, lhs, rhs, stream, num_sms);
"""

def m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(lhs: Tuple[torch.Tensor],
                                              rhs: Tuple[torch.Tensor],
                                              out: torch.Tensor, m_indices: torch.Tensor, configs = None) -> None:
    lhs = lhs
    rhs = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs.dtype == torch.bfloat16
    assert rhs.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, 1, num_sms, is_grouped_contiguous=True)
    expected_m = ceil_div(m, num_groups)
    extra_info = get_extra_info()
    args = (lhs, rhs, out,
            m_indices, m_indices, m, expected_m,
            torch.cuda.current_stream(), num_sms, smem_config[0], torch.empty(0).int())
    kernel_type = 'Default'
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_bf16_bf16_bf16_nt',
        keys={'N': n, 'K': k,
              'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n,
              'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
              'ENABLE_SBO_OVERLAP': False,
              'GEMM_TYPE': 'GroupedContiguous',
              'KERNEL_TYPE': kernel_type},
        space=(),
        includes=includes_cutlass3 if extra_info['use_cutlass3'] else includes ,
        arg_defs=(('lhs', torch.bfloat16),
                  ('rhs', torch.bfloat16),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('block_m_info', torch.int32),
                  ('m', int), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('signal', torch.int32)),
        template=template_cutlass3 if extra_info['use_cutlass3'] else template,
        jit_include_dir='cutlass3' if extra_info['use_cutlass3'] else None,
        args=args
    )

    # Run the kernel
    runtime(*args)


def m_grouped_gemm_bf16_bf16_bf16_nt_masked(lhs: Tuple[torch.Tensor],
                                            rhs: Tuple[torch.Tensor],
                                            out: torch.Tensor, masked_m: torch.Tensor, expected_m: int, configs = None,
                                            max_block_n: int = 256, enable_sbo_overlap: bool = False,
                                            signal: torch.Tensor = torch.empty(0).int()) -> None:
    num_groups, m, k = lhs.shape
    num_groups_, n, k_ = rhs.shape
    num_groups__, m_, n_ = out.shape
    num_groups___ = masked_m.numel()

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert lhs.dtype == torch.bfloat16
    assert rhs.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and masked_m.is_contiguous()

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
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms, is_grouped_masked=True, max_block_n=max_block_n)
    extra_info = get_extra_info()

    # Extra checks for TMA store
    # if num_groups > 1 and m > block_m:
    #     assert m % block_m == 0, f'For masked grouped GEMM, shape M should be multiple of the block M (current block M: {block_m})'

    args = (lhs, rhs, out,
            masked_m, masked_m, m, expected_m,
            torch.cuda.current_stream(), num_sms, smem_config[0], signal)
    kernel_type = 'Default'
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_bf16_bf16_bf16_nt',
        keys={'N': n, 'K': k,
              'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n,
              'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
              'ENABLE_SBO_OVERLAP': enable_sbo_overlap,
              'GEMM_TYPE': 'GroupedMasked',
              'KERNEL_TYPE': kernel_type},
        space=(),
        includes=includes_cutlass3 if extra_info['use_cutlass3'] else includes,
        arg_defs=(('lhs', torch.bfloat16),
                  ('rhs', torch.bfloat16),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('block_m_info', torch.int32), ('m', int), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('signal', torch.int32)),
        template=template_cutlass3 if extra_info['use_cutlass3'] else template,
        jit_include_dir='cutlass3' if extra_info['use_cutlass3'] else None,
        args=args
    )
    # Run the kernel
    runtime(*args)

    return (block_m, ceil_div(n, block_n))


def m_grouped_gemm_bf16_bf16_bf16_nt_fused(lhs: Tuple[torch.Tensor],
                                     rhs: Tuple[torch.Tensor],
                                     out: torch.Tensor,
                                     m_indices: torch.Tensor,
                                     m_rows: torch.Tensor = None,
                                     expert_ids: torch.Tensor = None,
                                     sorted_token_ids: torch.Tensor = None,
                                     block_m_offset: torch.Tensor = None,
                                     configs = None) -> None:
    """
    Perform a grouped GEMM (contiguous format) with BF16 inputs and BF16 output,.

    Requirements:
        LHS, RHS, and output tensors must be in contiguous format.
        RHS are required to be transposed.

    Arguments:
        lhs: the BF16 input tensor (typed `torch.bfloat16`) of shape `[num_token, k]`.
        rhs: the BF16 input tensor (typed `torch.bfloat16`) of shape `[num_groups, n, k]`.
        out: the BF16 output tensor of shape `[m_sum, n]`, representing the result.
        m_indices: topk_ids, a tensor of shape `[num_token, top_k]` with type `torch.int32`,
                indicating which expert each token selects.
        m_rows: a tensor of shape `[num_experts]`, indicating the number of tokens each expert processes.
        expert_ids: a tensor of shape `[num_blocks_m]` with type `torch.int32`,
                    indicating which expert each block processes.
        sorted_token_ids: a tensor of shape `[num_blocks_m, block_m]` with type `torch.int32`,
                        indicating the token indices each block processes,
                        padded with `num_token` for incomplete blocks.
        block_m_offset: a tensor of shape `[num_blocks_m]` with type `torch.int32`,
                        indicating the row offset in the output tensor for each block.

    Where:
        num_token: the number of input tokens.
        top_k: the number of experts selected per token.
        num_groups: the number of expert groups (equal to num_experts).
        num_experts: the total number of experts.
        k: the hidden dimension of input features.
        n: the hidden dimension of output features.
        block_m: the maximum number of tokens processed per block in m dimension.
        num_blocks_m: the total number of blocks in m dimension.
        m_sum: the total number of token-expert pairs, equal to `num_token * top_k`.
    """
    num_token, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_sum, n_ = out.shape
    m_sum_ = m_indices.numel()

    # Type and shape checks
    assert m_sum == m_sum_ and k == k_ and n == n_
    assert lhs.dtype == torch.bfloat16
    assert rhs.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()
    assert k % 8 == 0, "K must be a multiple of 8 so that 8 bfloat16 elements can be loaded with 128b aligned vectorized memory access."

    # Do nothing if `m_sum` is zero
    if m_sum == 0:
        return
     # Auto-tuning with compilation
    global includes_fusedmoe_gemm, template_fusedmoe_gemm
    num_sms = get_num_sms()
    # block_m, block_n, block_k, warp_m, warp_n, num_stages = 64, 128, 64, 32, 32, 2
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        expected_m = (m_sum + num_groups - 1) / num_groups
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms, is_grouped_contiguous=False)
    # 统计每个 expert 被选择的次数
    if m_rows is None:
        m_rows = torch.zeros(num_groups, dtype=torch.int32).to('cpu')
        for eid in m_indices.to('cpu'):
            m_rows[eid] += 1

    if expert_ids is None:
        expert_ids_v = []
        # for m in m_rows:
        for eid, m_per_expert in enumerate(m_rows):
            num_blocks_m_per_expert = (m_per_expert + block_m - 1) // block_m
            expert_ids_v.extend([eid] * num_blocks_m_per_expert)
        expert_ids = torch.tensor(expert_ids_v, dtype=torch.int32).to('cuda')

    num_blocks_m = expert_ids.shape[0]
    if sorted_token_ids is None:
        # 初始化 sorted_token_ids，用 num_token 填充
        sorted_token_ids = torch.full((num_blocks_m, block_m), num_token,
                                    dtype=torch.int32, device='cpu')
        # 预先计算每个 expert 的 token 列表
        expert_tokens = {}
        for expert_id in range(num_groups):
            # 找出哪些 token 选择了这个 expert
            mask = (m_indices == expert_id)  # [num_token, topk]
            selected = mask.any(dim=1)  # [num_token]
            expert_tokens[expert_id] = torch.where(selected)[0]  # [num_selected]
        # 记录每个 expert 已经处理了多少个 token（用于 block 切分）
        expert_token_pos = torch.zeros(num_groups, dtype=torch.int32, device='cpu')
        for block_idx in range(num_blocks_m):
            expert_id = expert_ids[block_idx].item()
            # 获取该 expert 的所有 token
            tokens = expert_tokens[expert_id]
            # 计算该 block 的起始位置
            start_pos = expert_token_pos[expert_id].item()
            # 计算该 block 处理的 token 范围
            end_pos = min(start_pos + block_m, len(tokens))
            # 复制 token 索引
            count = end_pos - start_pos
            sorted_token_ids[block_idx, :count] = tokens[start_pos:end_pos]
            # 更新该 expert 的 token 位置
            expert_token_pos[expert_id] += count
        sorted_token_ids = sorted_token_ids.to('cuda')

    if block_m_offset is None:
        """计算每个 block 的输出 offset"""
        # 计算每个 expert 的起始 offset（cumsum）
        expert_start_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device='cpu')
        expert_start_offsets[1:] = torch.cumsum(m_rows, dim=0)
        # 记录每个 expert 已经处理了多少个 block
        expert_block_count = torch.zeros(num_groups, dtype=torch.int32, device='cpu')
        # 计算每个 block 的 offset
        block_m_offset = torch.zeros(num_blocks_m, dtype=torch.int32, device='cpu')
        for block_idx in range(num_blocks_m):
            expert_id = expert_ids[block_idx].item()
            expert_offset = expert_start_offsets[expert_id].item()
            blk_num = expert_block_count[expert_id].item()
            block_offset_in_expert = blk_num * block_m
            block_m_offset[block_idx] = expert_offset + block_offset_in_expert
            expert_block_count[expert_id] += 1

        block_m_offset = block_m_offset.to('cuda')
    m_rows = m_rows.to('cuda')

    # print("m_indices:", m_indices.shape, m_indices)
    # print("m_rows:", m_rows.shape, m_rows)
    # print("expert_ids:", expert_ids.shape, expert_ids)
    # print("sorted_token_ids:", sorted_token_ids.shape, sorted_token_ids)
    # print("block_m_offset:", block_m_offset.shape, block_m_offset)

    # print("out_shape:", out.shape)
    # print("lhs_shape:", lhs.shape)
    ## the largest blockM_num is, num_groups - 1 only has 1 token, the last group has (m-1) tokens, blockM_num = num_group -1  + ceil_div(m + 1 - num_group, block_m)
    ## total line num: blockM_num + 1, line0 is used to store the real blockM_num
    ## total_size = (blockM_num + 1) * 4 * sizeof(int) Byte
    block_m_info = torch.empty((num_groups + ceil_div(m_sum + 1 - num_groups, block_m)) * 4, dtype=torch.int32, device=m_rows.device)
    args = (lhs, rhs, out, m_rows, block_m_info,
            sorted_token_ids, expert_ids, block_m_offset,
            num_token, torch.cuda.current_stream(), num_sms)

    kernel_type = 'Default'
    runtime = jit_tuner.compile_and_tune(
        name='fusedMoeGemm_bf16_bf16_bf16_nt',
        keys={'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_STAGES': num_stages,
                'ENABLE_SBO_OVERLAP': False,
                'GEMM_TYPE': 'GroupedFused',
                'KERNEL_TYPE': kernel_type},
        space=(),
        includes=includes_fusedmoe_gemm,
        arg_defs=(('lhs', torch.bfloat16),
                ('rhs', torch.bfloat16),
                ('out', torch.bfloat16),
                ('grouped_layout', torch.int32),
                ('block_m_info', torch.int32),
                ('sorted_token_ids', torch.int32),
                ('expert_ids', torch.int32),
                ('block_m_offset', torch.int32),
                ('m', int),
                ('stream', torch.cuda.Stream),
                ('num_sms', int)),
        template=template_fusedmoe_gemm,
        jit_include_dir='cutlass3',
        args=args
    )
    runtime(*args)

def m_grouped_gemm_bf16_bf16_bf16_nt_nopad(lhs: Tuple[torch.Tensor],
                                     rhs: Tuple[torch.Tensor],
                                     out: torch.Tensor, m_indices: torch.Tensor,
                                     m_rows: torch.Tensor = None, configs = None) -> None:
    lhs = lhs
    rhs = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs.dtype == torch.bfloat16
    assert rhs.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    expected_m = ceil_div(m, num_groups)

    # Do nothing if `m` is zero
    if m == 0:
        return

    if False:   # only used to debug fuse kernel path
        global includes_fusedmoe_gemm, template_fusedmoe_gemm
        num_sms = get_num_sms()
        if configs:
            num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
        else:
            num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms, is_grouped_contiguous=False)
        # 统计每个 expert 被选择的次数
        m_rows = torch.zeros(num_groups, dtype=torch.int32).to('cpu')
        for eid in m_indices.to('cpu'):
            m_rows[eid] += 1
        expert_ids_v = []
        # for m in m_rows:
        for eid, m_per_expert in enumerate(m_rows):
            num_blocks_m_per_expert = (m_per_expert + block_m - 1) // block_m
            expert_ids_v.extend([eid] * num_blocks_m_per_expert)
        expert_ids = torch.tensor(expert_ids_v, dtype=torch.int32).to('cuda')

        num_blocks_m = expert_ids.shape[0]

        """计算每个 block 的输出 offset"""
        # 计算每个 expert 的起始 offset（cumsum）
        expert_start_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device='cpu')
        expert_start_offsets[1:] = torch.cumsum(m_rows, dim=0)
        # 记录每个 expert 已经处理了多少个 block
        expert_block_count = torch.zeros(num_groups, dtype=torch.int32, device='cpu')
        # 计算每个 block 的 offset
        block_m_offset = torch.zeros(num_blocks_m, dtype=torch.int32, device='cpu')
        for block_idx in range(num_blocks_m):
            expert_id = expert_ids[block_idx].item()
            expert_offset = expert_start_offsets[expert_id].item()
            blk_num = expert_block_count[expert_id].item()
            block_offset_in_expert = blk_num * block_m
            block_m_offset[block_idx] = expert_offset + block_offset_in_expert
            expert_block_count[expert_id] += 1

        block_m_offset = block_m_offset.to('cuda')
        # 初始化 sorted_token_ids，用 m 填充
        sorted_token_ids = torch.full((num_blocks_m, block_m), m,
                                    dtype=torch.int32, device='cpu')

        for i, v in enumerate(torch.diff(block_m_offset)):
            for index in range(v):
                sorted_token_ids[i][index] = block_m_offset[i] + index
        for index in range(m - block_m_offset[-1]):
            sorted_token_ids[-1][index] = block_m_offset[-1] + index
        sorted_token_ids = sorted_token_ids.to('cuda')
        m_rows = m_rows.to('cuda')

        # print("block_m:", block_m)

        # print("m_indices:", m_indices.shape, m_indices)
        # print("m_rows:", m_rows.shape, m_rows)
        # print("expert_ids:", expert_ids.shape, expert_ids)
        # print("sorted_token_ids:", sorted_token_ids.shape, sorted_token_ids)
        # print("block_m_offset:", block_m_offset.shape, block_m_offset)

        # print("out_shape:", out.shape)
        # print("lhs_shape:", lhs.shape)
        ## the largest blockM_num is, num_groups - 1 only has 1 token, the last group has (m-1) tokens, blockM_num = num_group -1  + ceil_div(m + 1 - num_group, block_m)
        ## total line num: blockM_num + 1, line0 is used to store the real blockM_num
        ## total_size = (blockM_num + 1) * 4 * sizeof(int) Byte
        block_m_info = torch.empty((num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4, dtype=torch.int32, device=m_rows.device)
        args = (lhs, rhs, out, m_rows, block_m_info,
            sorted_token_ids, expert_ids, block_m_offset,
            m, torch.cuda.current_stream(), num_sms)

        ## Default, MultistageOnN, OverlapPrologue, OverlapMainloop
        kernel_type = 'Default'
        runtime = jit_tuner.compile_and_tune(
            name='fusedMoeGemm_bf16_bf16_bf16_nt',
            keys={'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                    'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                    'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_STAGES': num_stages,
                    'ENABLE_SBO_OVERLAP': False,
                    'GEMM_TYPE': 'GroupedFused',
                    'KERNEL_TYPE': kernel_type},
            space=(),
            includes=includes_fusedmoe_gemm,
            arg_defs=(('lhs', torch.bfloat16),
                    ('rhs', torch.bfloat16),
                    ('out', torch.bfloat16),
                    ('grouped_layout', torch.int32),
                    ('block_m_info', torch.int32),
                    ('sorted_token_ids', torch.int32),
                    ('expert_ids', torch.int32),
                    ('block_m_offset', torch.int32),
                    ('m', int),
                    ('stream', torch.cuda.Stream),
                    ('num_sms', int)),
            template=template_fusedmoe_gemm,
            jit_include_dir='cutlass3',
            args=args
        )
        runtime(*args)
        return

    # Auto-tuning with compilation
    global includes, template, includes_gemv, template_gemv
    num_sms = get_num_sms()
    use_gemv = False

    if ((k % 16 == 0 and ((m <= 2 * num_groups * 0.75 and k <= 32 * 8) or (m < 0.65 * num_groups and k > 256)) and not is_ppu1v5_device())
        or (m < 0.8 * num_groups and is_ppu1v5_device())):
        # use gemmv if avg m small
        # ThreadPerN = 8
        # NUM_UNROLL = 1
        # SWZL_SIZE_M = 1
        # NPerThread = 1
        BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread, USE_SMALL_K = get_gemv_best_configs(m, n, k, num_groups, num_sms, lhs.dtype)

        if ThreadPerN != -1:
            args = (lhs, rhs, out,
                m_indices, m,
                torch.cuda.current_stream())

            runtime = jit_tuner.compile_and_tune(
                name='m_grouped_gemv_bf16_bf16_bf16_nt',
                keys={'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                    'ThreadPerN':ThreadPerN, 'NUM_UNROLL':NUM_UNROLL,
                    'SWZL_SIZE_M':SWZL_SIZE_M, 'NPerThread':NPerThread,
                    'BlockSize':BlockSize, 'USE_SMALL_K':USE_SMALL_K,
                    'acc_type':'float'},
                space=(),
                includes=includes_gemv,
                arg_defs=(('lhs', torch.bfloat16),
                        ('rhs', torch.bfloat16),
                        ('out', torch.bfloat16),
                        ('grouped_layout', torch.int32), ('m', int),
                        ('stream', torch.cuda.Stream)),
                template=template_gemv,
                args=args
            )
            use_gemv = True

    if use_gemv == False:
        if configs:
            num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
        else:
            num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms, is_grouped_contiguous=False)
        extra_info = get_extra_info()

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

        args = (lhs, rhs, out,
                m_rows, block_m_info, m, expected_m,
                torch.cuda.current_stream(), num_sms, smem_config[0], torch.empty(0).int())

        ## Default, MultistageOnN, OverlapPrologue, OverlapMainloop
        kernel_type = 'Default'
        runtime = jit_tuner.compile_and_tune(
            name='m_grouped_gemm_bf16_bf16_bf16_nt',
            keys={'N': n, 'K': k,
                'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n,
                'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
                'ENABLE_SBO_OVERLAP': False,
                'GEMM_TYPE': 'GroupedNoPad',
                'KERNEL_TYPE': kernel_type
                },
            space=(),
            includes=includes_cutlass3 if extra_info['use_cutlass3'] else includes,
            arg_defs=(('lhs', torch.bfloat16),
                    ('rhs', torch.bfloat16),
                    ('out', torch.bfloat16),
                    ('grouped_layout', torch.int32), ('block_m_info', torch.int32), ('m', int), ('expected_m', int),
                    ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                    ('signal', torch.int32)),
            template=template_cutlass3 if extra_info['use_cutlass3'] else template,
            jit_include_dir='cutlass3' if extra_info['use_cutlass3'] else None,
            args=args
        )
    # Run the kernel
    runtime(*args)
