# Track training with Trackio

The default Runner records training, validation and test metrics in the local Trackio project `mmengine`. MMLogger writes progress and diagnostics to the terminal and the run's `.log` file.

Trackio requires Python 3.10 or newer. Earlier Python versions use the terminal and text log.

Epoch-based training uses the epoch as the validation step. Iteration-based training uses the iteration. Set `default_hooks.logger.log_metric_by_epoch` to choose the unit explicitly.

Each runner owns its tracking run. You can log images before training and call `runner.test()` after training to add results to that run. If you have already opened a Trackio run yourself, it stays active for your own logging.

To choose a project or run name, configure the visualizer in your training config.

```python
visualizer = dict(
    type='Visualizer',
    vis_backends=[dict(
        type='TrackioVisBackend',
        init_kwargs=dict(project='detector', name='baseline'))])
```

Run `trackio show --project detector` to view the metrics, configuration and images. Set `TRACKIO_DIR` before launching training and the dashboard to choose the local store. A local run requires no Hugging Face account.

When migrating an existing config, replace its list of built-in experiment backends with one Trackio entry and translate its project and run settings to `init_kwargs`. Historical files and dashboards can still be read with their original tools. [Trackio API](https://huggingface.co/docs/trackio/api)
