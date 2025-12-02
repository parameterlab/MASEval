# Callback

Callbacks allow you to hook into benchmark execution at various points. Use them for logging, monitoring, tracing, or custom side effects during agent runs.

::: maseval.core.callback.BenchmarkCallback

::: maseval.core.callback.EnvironmentCallback

::: maseval.core.callback.AgentCallback

## Built-in Callbacks

MASEval provides built-in callback implementations:

### Message Tracing

::: maseval.core.callbacks.message_tracing.MessageTracingAgentCallback

### Result Logging

::: maseval.core.callbacks.result_logger.ResultLogger

::: maseval.core.callbacks.result_logger.FileResultLogger

### Progress Bars

::: maseval.core.callbacks.progress_bar.ProgressBarCallback

::: maseval.core.callbacks.progress_bar.TqdmProgressBarCallback

::: maseval.core.callbacks.progress_bar.RichProgressBarCallback
