# Copyright (c) OpenMMLab. All rights reserved.
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.model import MMDistributedDataParallel
from mmengine.runner import loops


class BatchNormModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1, momentum=1.)
        self.register_buffer('local', torch.zeros(1))

    def forward(self, x):
        return self.bn(x)

    def val_step(self, x):
        self.output = self(x).item()
        return [dict(prediction=self.output)]

    test_step = val_step


def _evaluate_on_rank(rank, directory):
    dist.init_process_group(
        'gloo', init_method=(directory / 'world').as_uri(), rank=rank,
        world_size=2, timeout=timedelta(seconds=45))
    try:
        model = BatchNormModel()
        ddp = MMDistributedDataParallel(model, broadcast_buffers=False)
        runner = SimpleNamespace(model=ddp, call_hook=Mock())
        inputs = torch.tensor([[1.]])
        loader = DataLoader(inputs)
        for loop_type in (loops.ValLoop, loops.TestLoop):
            ddp.train()
            ddp(torch.tensor([[0.], [2.]]) + 2 * rank).sum().backward()
            assert model.bn.running_mean.item() == 1 + 2 * rank
            model.local.fill_(rank)
            if rank == 0:
                torch.save(model.state_dict(), directory / 'checkpoint.pth')
            dist.barrier()
            loop_type(runner, loader, Evaluator([])).run()
            assert model.local.item() == rank
            offline = BatchNormModel().eval()
            offline.load_state_dict(torch.load(directory / 'checkpoint.pth'))
            assert model.output == offline(inputs).item()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                    reason='Gloo is required')
def test_evaluation_uses_checkpoint_batchnorm_buffers(tmp_path):
    torch.multiprocessing.spawn(_evaluate_on_rank, (tmp_path,), nprocs=2)
