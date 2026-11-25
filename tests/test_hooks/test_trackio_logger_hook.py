# Copyright (c) OpenMMLab. All rights reserved.
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mmengine.hooks import TrackioLoggerHook
from mmengine.registry import HOOKS
from mmengine.testing import RunnerTestCase
from mmengine.visualization import TrackioVisBackend

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Trackio requires Python 3.10+")


class TestTrackioLoggerHook(RunnerTestCase):

    def test_runner(self):
        cfg = self.epoch_based_cfg
        cfg.train_cfg.max_epochs = 1
        cfg.log_processor = dict(log_with_hierarchy=True)
        cfg.default_hooks.logger = dict(
            type="TrackioLoggerHook",
            interval=2,
            init_kwargs=dict(project="unit-test"))
        sdk = MagicMock()
        run = sdk.init.return_value
        sdk.context_vars.current_run.get.return_value = run
        sdk.finish.side_effect = run.finish
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setitem(sys.modules, "trackio", sdk)
            runner = self.build_runner(cfg)
            self.assertIsInstance(
                runner.visualizer.get_backend("TrackioVisBackend"),
                TrackioVisBackend)
            runner.train()
            sdk.init.assert_called_once_with(
                project="unit-test", name=runner.experiment_name)
            train_steps = [
                event.kwargs["step"] for event in run.log.call_args_list
                if "train/loss" in event.args[0]
            ]
            self.assertEqual(train_steps, [2, 4])
            val_steps = [
                event.kwargs["step"] for event in run.log.call_args_list
                if "val/acc" in event.args[0]
            ]
            self.assertEqual(val_steps, [1])
            self.assertTrue(list(Path(runner.log_dir).glob("*.log")))
            self.assertFalse(
                (Path(runner.log_dir) / "vis_data" / "scalars.json").exists())
            self.assertIn("unit-test", sdk.Markdown.call_args.args[0])
            run.finish.assert_called_once_with()

            run.reset_mock()
            metrics = runner.test()
            test_event = next(event for event in run.log.call_args_list
                              if "test/acc" in event.args[0])
            self.assertEqual(test_event.kwargs["step"], runner.iter)
            for key, value in metrics.items():
                self.assertEqual(test_event.args[0][f"test/{key}"], value)
            self.assertIn("test/time", test_event.args[0])
            run.finish.assert_called_once_with()

    def test_named_backend_accepts_preview_before_training(self):
        cfg = self.epoch_based_cfg
        cfg.train_cfg.max_epochs = 1
        cfg.visualizer = dict(
            type='Visualizer',
            vis_backends=[dict(type='TrackioVisBackend', name='dashboard')])
        cfg.default_hooks.logger = dict(
            type='TrackioLoggerHook',
            interval=1,
            init_kwargs=dict(project='previews'))
        sdk = MagicMock()
        with patch.dict(sys.modules, trackio=sdk):
            runner = self.build_runner(cfg)
            runner.visualizer.add_scalar('preview_width', 32)
            runner.train()
        assert sdk.init.call_count == 1
        assert sdk.init.call_args.kwargs["name"] == runner.experiment_name
        assert sdk.init.call_args.kwargs['project'] == 'previews'
        assert runner.visualizer.get_backend('dashboard') is not None

    def test_default_backend_configured_before_preview(self):
        cfg = self.epoch_based_cfg
        cfg.train_cfg.max_epochs = 1
        cfg.default_hooks.logger = dict(
            type='TrackioLoggerHook', init_kwargs=dict(project='previews'))
        sdk = MagicMock()
        with patch.dict(sys.modules, trackio=sdk):
            runner = self.build_runner(cfg)
            runner.visualizer.add_scalar('preview_width', 32)
            runner.train()
        sdk.init.assert_called_once_with(
            project='previews', name=runner.experiment_name)

    def test_iteration_validation_steps(self):
        cfg = self.iter_based_cfg
        cfg.train_cfg.update(max_iters=2, val_interval=1)
        cfg.log_processor = dict(by_epoch=False, log_with_hierarchy=True)
        cfg.default_hooks.logger = dict(type='LoggerHook', interval=1)
        sdk = MagicMock()
        with patch.dict(sys.modules, trackio=sdk):
            self.build_runner(cfg).train()
        steps = [
            call.kwargs['step']
            for call in sdk.init.return_value.log.call_args_list
            if 'val/acc' in call.args[0]
        ]
        assert steps == [1, 2]

    def test_non_main_process(self):
        hook = HOOKS.build(dict(type="TrackioLoggerHook"))
        self.assertIsInstance(hook, TrackioLoggerHook)
        runner = MagicMock(timestamp="worker")
        with patch(
                "mmengine.hooks.trackio_logger_hook.is_main_process",
                return_value=False):
            hook.before_register(runner)
            hook.before_run(runner)
        runner.visualizer.get_backend.assert_not_called()
