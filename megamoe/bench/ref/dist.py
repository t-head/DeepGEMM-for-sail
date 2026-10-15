"""Distributed bring-up and collective helpers for multi-rank test runs."""

import faulthandler
import inspect
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

_local_rank = None


def init_dist(local_rank: int, num_local_ranks: int) -> Tuple[int, int, dist.ProcessGroup]:
    """Source-compatible distributed initialization for local/multi-node tests."""
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    global _local_rank
    _local_rank = local_rank
    if dist.is_initialized():
        torch.set_default_device('cuda')
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), dist.group.WORLD
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in inspect.signature(dist.init_process_group).parameters:
        params['device_id'] = torch.device(f'cuda:{local_rank}')
    # Bounded rendezvous: a stuck PG bring-up (eager connect carries no timeout of its own) must
    # fail -- with a thread dump -- and let the caller retry, instead of hanging forever.
    init_timeout_sec = float(os.getenv('DG_DIST_INIT_TIMEOUT_SEC', '120'))
    if init_timeout_sec > 0:
        faulthandler.dump_traceback_later(init_timeout_sec, exit=True)
    try:
        dist.init_process_group(**params)
    finally:
        if init_timeout_sec > 0:
            faulthandler.cancel_dump_traceback_later()
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)
    group = dist.group.WORLD
    return dist.get_rank(), dist.get_world_size(), group


def uneven_all_gather(tensor: torch.Tensor, dim: int = 0, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    local_size = torch.tensor([tensor.shape[dim]], device=tensor.device, dtype=torch.long)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size, group=group)
    all_sizes = [int(size.item()) for size in all_sizes]
    max_size = max(all_sizes)

    if tensor.shape[dim] < max_size:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = max_size - tensor.shape[dim]
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=dim)
    else:
        tensor = tensor.contiguous()

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat([torch.narrow(value, dim, 0, all_sizes[index]) for index, value in enumerate(gathered)], dim=dim)


def dist_print(message: str = '', once_in_node: bool = False) -> None:
    if _local_rank is None:
        raise RuntimeError("init_dist must be called before dist_print")
    if not once_in_node or _local_rank == 0:
        print(message, flush=True)
    dist.barrier()
