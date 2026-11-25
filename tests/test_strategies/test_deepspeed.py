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


@pytest.mark.parametrize('legacy', [False, True])
def test_register_available_optimizers_once(monkeypatch, legacy):
    import sys
    from types import ModuleType

    from mmengine._strategy.deepspeed import register_deepspeed_optimizers
    from mmengine.registry import OPTIMIZERS

    modules = {
        'deepspeed.ops.adam': ['DeepSpeedCPUAdam', 'FusedAdam'],
        'deepspeed.ops.lamb': ['FusedLamb'],
        'deepspeed.runtime.fp16.onebit':
        ['OnebitAdam', 'OnebitLamb', 'ZeroOneAdam']
    }
    names = []
    monkeypatch.setitem(sys.modules, 'deepspeed', ModuleType('deepspeed'))
    for module_name, class_names in modules.items():
        module = ModuleType(module_name)
        for name in class_names:
            setattr(module, name, type(name, (), {}))
        if module_name.endswith('.onebit') and not legacy:
            module = None
        else:
            names.extend(class_names)
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(OPTIMIZERS, '_module_dict', {})
    assert register_deepspeed_optimizers() == names
    first = OPTIMIZERS.module_dict.copy()
    assert register_deepspeed_optimizers() == names
    assert OPTIMIZERS.module_dict == first
