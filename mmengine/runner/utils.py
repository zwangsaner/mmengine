# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import random
import time
from collections import Counter
from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmengine.device import is_cuda_available, is_musa_available
from mmengine.dist import broadcast_object_list, get_rank, sync_random_seed
from mmengine.logging import print_log
from mmengine.utils import digit_version, is_list_of
from mmengine.utils.dl_utils import TORCH_VERSION

_timestamp_counts = Counter()
_timestamp_lock = Lock()


def _reserve_run_timestamp(work_dir: str) -> str:
    """Reserve one log directory on rank zero and share its name with all ranks."""
    result: List[Union[str, OSError]] = ['']
    if get_rank() == 0:
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        try:
            with _timestamp_lock:
                while True:
                    index = _timestamp_counts[stamp]
                    _timestamp_counts[stamp] += 1
                    name = f'{stamp}_{index:03d}' if index else stamp
                    try:
                        os.mkdir(os.path.join(work_dir, name))
                    except FileExistsError:
                        continue
                    result[0] = name
                    break
        except OSError as exc:
            # Let every worker leave the collective if the destination fails.
            result[0] = exc
    broadcast_object_list(result)
    timestamp = result[0]
    if isinstance(timestamp, OSError):
        raise timestamp
    return timestamp


def calc_dynamic_intervals(
    start_interval: int,
    dynamic_interval_list: Optional[List[Tuple[int, int]]] = None
) -> Tuple[List[int], List[int]]:
    """Calculate dynamic intervals.

    Args:
        start_interval (int): The interval used in the beginning.
        dynamic_interval_list (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.

    Returns:
        Tuple[List[int], List[int]]: a list of milestone and its corresponding
        intervals.
    """
    if dynamic_interval_list is None:
        return [0], [start_interval]

    assert is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


def set_random_seed(seed: Optional[int] = None,
                    deterministic: bool = False,
                    diff_rank_seed: bool = False) -> int:
    """Set random seed.

    Args:
        seed (int, optional): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Defaults to False.
        diff_rank_seed (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Defaults to False.
    """
    if seed is None:
        seed = sync_random_seed()

    if diff_rank_seed:
        rank = get_rank()
        seed += rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    if is_cuda_available():
        torch.cuda.manual_seed_all(seed)
    elif is_musa_available():
        torch.musa.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print_log(
                'torch.backends.cudnn.benchmark is going to be set as '
                '`False` to cause cuDNN to deterministically select an '
                'algorithm',
                logger='current',
                level=logging.WARNING)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if digit_version(TORCH_VERSION) >= digit_version('1.10.0'):
            torch.use_deterministic_algorithms(True)
    return seed


def _get_batch_size(dataloader: dict):
    if isinstance(dataloader, dict):
        if 'batch_size' in dataloader:
            return dataloader['batch_size']
        elif ('batch_sampler' in dataloader
              and 'batch_size' in dataloader['batch_sampler']):
            return dataloader['batch_sampler']['batch_size']
        else:
            raise ValueError('Please set batch_size in `Dataloader` or '
                             '`batch_sampler`')
    elif isinstance(dataloader, DataLoader):
        return dataloader.batch_sampler.batch_size
    else:
        raise ValueError('dataloader should be a dict or a Dataloader '
                         f'instance, but got {type(dataloader)}')
