try:
    from .._C import (
        get_tma_aligned_size,
        get_mn_major_tma_aligned_tensor,
    )
except ImportError:
    # Expected behavior for HGGC runtime version before 12.1
    pass

from .._C import (
    set_mk_alignment_for_contiguous_layout,
    get_mk_alignment_for_contiguous_layout,
    get_theoretical_mk_alignment_for_contiguous_layout,
)

# Some alias
get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
