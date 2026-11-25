# Copyright (c) OpenMMLab. All rights reserved.
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mmengine.registry import VISBACKENDS
from mmengine.visualization import TrackioVisBackend

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Trackio requires Python 3.10+")


def test_trackio_backend(tmp_path, monkeypatch):
    sdk = MagicMock()
    monkeypatch.setitem(sys.modules, "trackio", sdk)
    backend = TrackioVisBackend(str(tmp_path))
    backend.close()
    sdk.init.assert_not_called()
    run = backend.experiment
    sdk.init.assert_called_once_with(project="mmengine")
    assert run is sdk.init.return_value

    metrics = {
        "loss": torch.tensor(0.5),
        "count": np.array([3]),
        "ratio": np.float32(0.25),
        "invalid": object()
    }
    with pytest.warns(UserWarning, match="invalid"):
        backend.add_scalars(metrics, step=12)
    run.log.assert_called_once_with({
        "loss": 0.5,
        "count": 3,
        "ratio": 0.25
    },
                                    step=12)
    assert isinstance(metrics["count"], np.ndarray)
    with pytest.raises(ValueError, match="step"):
        backend.add_scalars({"step": 99})
    backend.add_scalar("accuracy", 0.75, step=2)
    run.log.assert_called_with({"accuracy": 0.75}, step=2)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    backend.add_image("sample", image, step=2)
    sdk.Image.assert_called_once_with(image)
    run.log.assert_called_with({"sample": sdk.Image.return_value}, step=2)
    backend.close()
    backend.close()
    run.finish.assert_called_once_with()


def test_missing_trackio_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "trackio", None)
    backend = TrackioVisBackend(str(tmp_path))
    with pytest.raises(ImportError, match="pip install trackio"):
        backend.add_scalar("loss", 1)


def test_only_trackio_is_registered():
    assert VISBACKENDS.get("TrackioVisBackend") is TrackioVisBackend
    for name in ("Local", "Wandb", "Tensorboard", "MLflow", "ClearML",
                 "Neptune", "DVCLive", "Aim"):
        assert VISBACKENDS.get(name + "VisBackend") is None


def _record_local_experiment():
    from pathlib import Path

    from trackio.sqlite_storage import SQLiteStorage

    from mmengine.testing import RunnerTestCase

    case = RunnerTestCase()
    case.setUp()
    try:
        cfg = case.epoch_based_cfg
        cfg.train_cfg.max_epochs = 1
        cfg.log_processor = dict(log_with_hierarchy=True)
        runner = case.build_runner(cfg)
        runner.visualizer.add_image('preview',
                                    np.zeros((3, 4, 3), dtype=np.uint8))
        runner.train()
        runner.test()
        runs = SQLiteStorage.get_runs('mmengine')
        assert len(runs) == 1
        rows = SQLiteStorage.get_logs('mmengine', runs[0])
        assert any('config' in row for row in rows)
        assert any('preview' in row for row in rows)
        assert any('train/loss' in row and row['step'] > 0 for row in rows)
        assert any(row.get('val/acc') == 1 for row in rows)
        assert any(
            row.get("test/acc") == 1 and "test/time" in row for row in rows)
        assert list(Path(runner.log_dir).glob('*.log'))
        assert not (Path(runner.log_dir) / 'vis_data' /
                    'scalars.json').exists()
    finally:
        case.tearDown()


def test_trackio_local_experiment(tmp_path, monkeypatch):
    import multiprocessing

    monkeypatch.setenv('TRACKIO_DIR', str(tmp_path))
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    for key in ('TRACKIO_SPACE_ID', 'TRACKIO_SERVER_URL', 'TRACKIO_DATASET_ID',
                'TRACKIO_BUCKET_ID'):
        monkeypatch.delenv(key, raising=False)
    process = multiprocessing.get_context('spawn').Process(
        target=_record_local_experiment)
    process.start()
    process.join(90)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail('The local Trackio run did not finish')
    assert process.exitcode == 0
