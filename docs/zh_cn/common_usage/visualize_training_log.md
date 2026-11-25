# 使用 Trackio 查看训练日志

Runner 默认将训练、验证和测试指标写入本地 Trackio 项目 `mmengine`。MMLogger 将进度和诊断信息写入终端和运行目录的 `.log` 文件。

Trackio 要求 Python 3.10 或更高版本。较早的 Python 版本使用终端和文本日志。

在配置文件中可以指定项目名和运行名。

```python
visualizer = dict(
    type='Visualizer',
    vis_backends=[dict(
        type='TrackioVisBackend',
        init_kwargs=dict(project='detector', name='baseline'))])
```

运行 `trackio show --project detector` 查看指标、配置和图片。在启动训练和面板前设置 `TRACKIO_DIR` 可以指定本地存储目录。本地记录无需 Hugging Face 账号。

迁移旧配置时，将原有内置后端列表替换为一个 Trackio 后端，并通过 `init_kwargs` 设置项目和运行名称。历史文件仍可用原来的工具读取。[Trackio API](https://huggingface.co/docs/trackio/api)
