import torch
from typing import Tuple

from .gemm import get_best_configs, get_gemv_best_configs
from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_case_id
import os

# C++ code templates
includes = ('"deep_gemm/fp16_gemm.cuh"', )
includes_cutlass3 = ('"../deep_gemm/fp16_gemm_cutlass3.cuh"', )
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

// Make a templated grouped GEMM
using gemm_t = Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}>;

// Launch kernel
gemm_t::run(out, grouped_layout,
            m, expected_m, lhs, rhs,
            stream, num_sms, smem_size);
"""

includes_gemv = ('"deep_gemm/gemvt.cuh"', )
template_gemv = """
using namespace deep_gemm;

// Templated args from Python JIT call
using D = __nv_bfloat16;
constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto ThreadPerN = {ThreadPerN};
constexpr auto NPerThread = {NPerThread};
constexpr auto NUM_UNROLL = {NUM_UNROLL};
constexpr auto SWZL_SIZE_M = {SWZL_SIZE_M};
constexpr auto USE_SMALL_K = {USE_SMALL_K};
constexpr auto BlockSize = {BlockSize};

// Make a templated grouped GEMM
using gemm_v = Gemvt<D, D, N, K, kNumGroups, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M, BlockSize, USE_SMALL_K>;

// Launch kernel
gemm_v::run(out, grouped_layout,
            m, lhs, rhs,
            stream);
"""

def m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(lhs: Tuple[torch.Tensor],
                                              rhs: Tuple[torch.Tensor],
                                              out: torch.Tensor, m_indices: torch.Tensor) -> None:
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
    num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config, extra_info = get_best_configs(m, n, k, 1, num_sms, is_grouped_contiguous=True)

    expected_m = 0
    args = (lhs, rhs, out,
            m_indices, m, expected_m, num_groups,
            torch.cuda.current_stream(), num_sms, smem_config[0])
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_bf16_bf16_bf16_nt',
        keys={'N': n, 'K': k,
              'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n,
              'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
              'GEMM_TYPE': 'GroupedContiguous'},
        space=(),
        includes=includes_cutlass3 if extra_info['use_cutlass3'] else includes ,
        arg_defs=(('lhs', torch.bfloat16),
                  ('rhs', torch.bfloat16),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('m', int), 
                  ('num_groups', int), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        jit_include_dir='cutlass3' if extra_info['use_cutlass3'] else None,
        args=args
    )

    # Run the kernel
    runtime(*args)


def m_grouped_gemm_bf16_bf16_bf16_nt_masked(lhs: Tuple[torch.Tensor],
                                            rhs: Tuple[torch.Tensor],
                                            out: torch.Tensor, masked_m: torch.Tensor, expected_m: int) -> None:
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

    # Auto-tuning with compilation
    global includes, template

    num_sms = get_num_sms()
    num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config, extra_info = get_best_configs(expected_m, n, k, num_groups, num_sms, is_grouped_masked=True)

    # Extra checks for TMA store
    if num_groups > 1 and m > block_m:
        assert m % block_m == 0, f'For masked grouped GEMM, shape M should be multiple of the block M (current block M: {block_m})'

    args = (lhs, rhs, out,
            masked_m, m, expected_m,
            torch.cuda.current_stream(), num_sms, smem_config[0])

    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_bf16_bf16_bf16_nt',
        keys={'N': n, 'K': k,
              'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n,
              'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
              'GEMM_TYPE': 'GroupedMasked'},
        space=(),
        includes=includes_cutlass3 if extra_info['use_cutlass3'] else includes,
        arg_defs=(('lhs', torch.bfloat16),
                  ('rhs', torch.bfloat16),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('m', int), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        jit_include_dir='cutlass3' if extra_info['use_cutlass3'] else None,
        args=args
    )

    # Run the kernel
    runtime(*args)


def m_grouped_gemm_bf16_bf16_bf16_nt_nopad(lhs: Tuple[torch.Tensor],
                                     rhs: Tuple[torch.Tensor],
                                     out: torch.Tensor, m_indices: torch.Tensor,
                                     m_rows: torch.Tensor = None) -> None:
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

    # Auto-tuning with compilation
    global includes, template, includes_gemv, template_gemv
    num_sms = get_num_sms()
    use_gemv = False


    if expected_m <= 2 and (k % 64 == 0 or (n >= 1024 and k <= 32 * 8)):
        # use gemmv if avg m small
        # ThreadPerN = 8
        # NUM_UNROLL = 1
        # SWZL_SIZE_M = 1
        # NPerThread = 1
        BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread, USE_SMALL_K = get_gemv_best_configs(m, n, k, num_groups, num_sms)

        if ThreadPerN != -1:
            args = (lhs, rhs, out,
                m_indices, m,
                torch.cuda.current_stream())

            runtime = jit_tuner.compile_and_tune(
                name='m_grouped_gemv_bf16_bf16_bf16_nt',
                keys={'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                    'ThreadPerN':ThreadPerN, 'NUM_UNROLL':NUM_UNROLL,
                    'SWZL_SIZE_M':SWZL_SIZE_M, 'NPerThread':NPerThread,
                    'BlockSize':BlockSize, 'USE_SMALL_K':USE_SMALL_K},
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
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config, extra_info = get_best_configs(expected_m, n, k, num_groups, num_sms, is_grouped_contiguous=False)

        if m_rows is None:
            experts_for_rows = torch.zeros(num_groups + 1, dtype=torch.int32, device='cuda')
            counts = torch.bincount(m_indices)
            min_n = min(counts.size(0), num_groups)
            if min_n > 0:
                experts_for_rows[1:1+min_n] = counts[:min_n]
            m_rows = experts_for_rows.cumsum(0)

        args = (lhs, rhs, out,
                m_rows, m, expected_m,
                torch.cuda.current_stream(), num_sms, smem_config[0])

        runtime = jit_tuner.compile_and_tune(
            name='m_grouped_gemm_bf16_bf16_bf16_nt',
            keys={'N': n, 'K': k,
                'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n,
                'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
                'GEMM_TYPE': 'GroupedNoPad'},
            space=(),
            includes=includes,
            arg_defs=(('lhs', torch.bfloat16),
                    ('rhs', torch.bfloat16),
                    ('out', torch.bfloat16),
                    ('grouped_layout', torch.int64), ('m', int), ('expected_m', int),
                    ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
            template=template,
            args=args
        )

    dump_env = os.getenv('dump_group_m')
    if dump_env:
        filename = f"case{get_case_id()}_groups{num_groups}_m{m}_n{n}_k{k}_em{expected_m}_GroupedNoPad.dump"
        tensor_cpu = m_indices.detach().cpu()
        data = tensor_cpu.tolist()

        print(f"[INFO] file:{filename} with size:{m_indices.size()}\n")

        with open(filename, 'w', encoding='utf-8') as f:
            for num in data:
                f.write(f"{num}\n")

    # Run the kernel
    runtime(*args)