# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmengine.dist import is_main_process
from mmengine.registry import HOOKS, VISBACKENDS
from .logger_hook import LoggerHook


@HOOKS.register_module()
class TrackioLoggerHook(LoggerHook):
    """Log a runner's metrics and visualizations to a native Trackio run.

    Configure this hook as ``default_hooks.logger``. It attaches a Trackio
    backend to the runner's visualizer on the main process. After
    ``before_run``, custom hooks can access the same run through
    ``runner.visualizer.get_backend("TrackioVisBackend").experiment``.

    Args:
        init_kwargs (dict, optional): Arguments for ``trackio.init``. The
            project defaults to ``mmengine`` and the run name defaults to
            ``runner.experiment_name``.
        **kwargs: Logging options accepted by :class:`LoggerHook`.
    """

    def __init__(self, init_kwargs: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self._init_kwargs = dict(init_kwargs or {})

    def before_run(self, runner) -> None:
        """Attach Trackio before the first visualization event of the loop."""
        if is_main_process():
            backend = runner.visualizer.get_backend("TrackioVisBackend")
            if backend is None:
                backend = VISBACKENDS.build(
                    dict(
                        type="mmengine.TrackioVisBackend",
                        save_dir=runner.log_dir,
                        init_kwargs=dict(name=runner.experiment_name)))
                runner.visualizer.add_backend(backend)
            backend.configure(self._init_kwargs)
        super().before_run(runner)
