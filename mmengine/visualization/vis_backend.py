# Copyright (c) OpenMMLab. All rights reserved.
import functools
import json
import logging
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch

from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.registry import VISBACKENDS


def force_init_env(old_func: Callable) -> Any:
    """Those methods decorated by ``force_init_env`` will be forced to call
    ``_init_env`` if the instance has not been fully initiated. This function
    will decorated all the `add_xxx` method and `experiment` method, because
    `VisBackend` is initialized only when used its API.

    Args:
        old_func (Callable): Decorated function, make sure the first arg is an
            instance with ``_init_env`` method.

    Returns:
        Any: Depends on old_func.
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `_init_env` method.
        if not hasattr(obj, '_init_env'):
            raise AttributeError(f'{type(obj)} does not have _init_env '
                                 'method.')
        # If instance does not have `_env_initialized` attribute or
        # `_env_initialized` is False, call `_init_env` and set
        # `_env_initialized` to True
        if not getattr(obj, '_env_initialized', False):
            print_log(
                'Attribute `_env_initialized` is not defined in '
                f'{type(obj)} or `{type(obj)}._env_initialized is '
                'False, `_init_env` will be called and '
                f'{type(obj)}._env_initialized will be set to True',
                logger='current',
                level=logging.DEBUG)
            obj._init_env()  # type: ignore
            obj._env_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper


class BaseVisBackend(metaclass=ABCMeta):
    """Base class for visualization backend.

    All backends must inherit ``BaseVisBackend`` and implement
    the required functions.

    Args:
        save_dir (str, optional): The root directory to save
            the files produced by the backend.
    """

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        self._env_initialized = False

    @property
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this visualization
        backend.

        Use the native run to record additional media such as tables.
        """
        pass

    @abstractmethod
    def _init_env(self) -> Any:
        """Setup env for VisBackend."""
        pass

    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config.

        Args:
            config (Config): The Config object
        """
        pass

    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record the model graph.

        Args:
            model (torch.nn.Module): Model to draw.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        pass

    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        pass

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar.

        Args:
            name (str): The scalar identifier.
            value (int, float): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        pass

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): The scalar's data will be
                saved to the `file_path` file at the same time
                if the `file_path` parameter is specified.
                Defaults to None.
        """
        pass

    def close(self) -> None:
        """Close an opened object."""
        pass


@VISBACKENDS.register_module()
class TrackioVisBackend(BaseVisBackend):
    """Send scalars, RGB images and configuration reports to a Trackio run.

    Args:
        save_dir (str): MMEngine visualization directory. Trackio uses its
            own local store, configurable with the ``TRACKIO_DIR`` environment
            variable.
        init_kwargs (dict, optional): Arguments for ``trackio.init``.
            The default project is ``mmengine``.
    """

    def __init__(self, save_dir: str, init_kwargs: Optional[dict] = None):
        super().__init__(save_dir)
        self._init_kwargs = {"project": "mmengine", **(init_kwargs or {})}
        self._resume_name = None

    def configure(self, init_kwargs: dict) -> None:
        """Apply runner options before starting a Trackio run."""
        if any(self._init_kwargs.get(k) != v for k, v in init_kwargs.items()):
            if self._env_initialized:
                raise RuntimeError("The Trackio run has already started.")
            self._init_kwargs.update(init_kwargs)
            self._resume_name = None

    def _init_env(self):
        try:
            import trackio
        except ModuleNotFoundError as exc:
            if exc.name != "trackio":
                raise
            raise ImportError(
                'Install Trackio with "pip install trackio" to use TrackioVisBackend.'
            ) from exc

        kwargs = self._init_kwargs.copy()
        if self._resume_name is not None:
            kwargs.update(name=self._resume_name, resume="allow")
        self._run = trackio.init(**kwargs)
        self._trackio = trackio

    @property
    @force_init_env
    def experiment(self):
        """Return the native Trackio run for logging tables, artifacts and other media."""
        return self._run

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the MMEngine configuration as a readable report."""
        report = self._trackio.Markdown(
            f"```python\n{config.pretty_text}\n```")
        self._run.log({"config": report}, step=0)

    @force_init_env
    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record the model architecture in the run."""
        self._run.log(
            {"model": self._trackio.Markdown(f"```text\n{model}\n```")},
            step=0)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Log an RGB image at its MMEngine step."""
        self._run.log({name: self._trackio.Image(image)}, step=step)

    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Log a single numeric metric."""
        self.add_scalars({name: value}, step=step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Log one batch of numeric metrics at the step supplied by LoggerHook."""
        if "step" in scalar_dict:
            raise ValueError("Pass step as the step argument.")
        metrics = {}
        for name, value in scalar_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.item()
            elif isinstance(value,
                            (np.ndarray, np.generic)) and value.size == 1:
                value = value.item()
            if isinstance(value, (np.ndarray, torch.Tensor)):
                value = value.tolist()
            if isinstance(value, (dict, list, tuple)):
                # Preserve structured evaluation results in the run as a readable report.
                metrics[name] = self._trackio.Markdown(
                    "```json\n" + json.dumps(value, ensure_ascii=False) +
                    "\n```")
            elif isinstance(value, (int, float)):
                metrics[name] = value
            elif isinstance(value, str):
                metrics[name] = self._trackio.Markdown(value)
            else:
                warnings.warn(
                    f"Skipping non-scalar Trackio metric {name!r} of type {type(value).__name__}."
                )
        if metrics:
            self._run.log(metrics, step=step)

    def close(self) -> None:
        """Flush the run if this backend started one."""
        if self._env_initialized:
            # Let the SDK clean up its active run when this backend owns it.
            if self._trackio.context_vars.current_run.get() is self._run:
                self._trackio.finish()
            else:
                self._run.finish()
            self._resume_name = self._run.name
            self._env_initialized = False
