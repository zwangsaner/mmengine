# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from mmengine._strategy.deepspeed import DeepSpeedStrategy


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('version', ['0.12.0', '0.14.0'])
def test_load_checkpoint_options(version, strict):
    engine = Mock()
    engine.load_checkpoint.return_value = ('dir/tag', {'epoch': 2})
    strategy = SimpleNamespace(model=engine, logger=Mock(),
                               exclude_frozen_parameters=False)
    with patch('mmengine._strategy.deepspeed.deepspeed',
               SimpleNamespace(__version__=version)):
        state = DeepSpeedStrategy.load_checkpoint(strategy, 'dir/tag',
                                                  strict=strict)
    assert state == {'epoch': 2}
    engine.load_checkpoint.assert_called_once_with(
        'dir', tag='tag', load_module_strict=strict, load_optimizer_states=False,
        load_module_only=True)
