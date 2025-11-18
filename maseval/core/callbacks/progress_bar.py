"""Progress bar callbacks for benchmark execution.

This module provides progress tracking for benchmark runs using different progress bar libraries.
Implementations for both tqdm and rich are provided, with tqdm as the default.
"""

import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

from ..callback import BenchmarkCallback

if TYPE_CHECKING:
    from ..benchmark import Benchmark


class ProgressBarCallback(BenchmarkCallback, ABC):
    """Abstract base class for progress bar callbacks.

    Displays benchmark execution progress including overall completion, success rate,
    time elapsed/remaining, and custom metrics. Automatically tracks benchmark execution
    and updates the progress bar as tasks complete.

    Use `TqdmProgressBarCallback` or `RichProgressBarCallback` directly, or subclass them
    to customize metric display.

    **User-facing methods:**

    - `set_metrics(**metrics)`: Manually update displayed metrics
    - `update_metrics(report)`: Override to automatically extract metrics from task reports

    Example:
        ```python
        from maseval.core.callbacks.progress_bar import TqdmProgressBarCallback

        # Option 1: Use directly with manual metric updates
        progress_bar = TqdmProgressBarCallback()
        benchmark = MyBenchmark(callbacks=[progress_bar])
        benchmark.run(tasks)
        progress_bar.set_metrics(accuracy="95.2%", avg_score="0.87")

        # Option 2: Subclass to automatically extract metrics from reports
        class MyProgressBar(TqdmProgressBarCallback):
            def update_metrics(self, report):
                if "evaluation_result" in report:
                    return {"accuracy": f"{report['evaluation_result']['acc']:.1%}"}
                return {}

        progress_bar = MyProgressBar()
        benchmark = MyBenchmark(callbacks=[progress_bar])
        benchmark.run(tasks)  # Metrics auto-update after each task
        ```

    Args:
        desc: Custom description. Defaults to "Running {BenchmarkClassName}"
        show_status: Whether to display success counter (X/Y Successful)
    """

    def __init__(self, desc: Optional[str] = None, show_status: bool = True):
        super().__init__()
        self.desc = desc
        self.show_status = show_status
        self._lock = threading.Lock()
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset progress tracking state."""
        self.total = 0
        self.current = 0
        self.successful = 0
        self._custom_metrics: Dict[str, str] = {}

    def on_run_start(self, benchmark: "Benchmark") -> None:
        """Called by benchmark framework when run starts."""
        # Reset state for fresh run (supports calling run() multiple times)
        self._reset_state()

        # Calculate total iterations: tasks × repeats
        n_tasks = len(list(benchmark.tasks))
        n_repeats = benchmark.n_task_repeats
        self.total = n_tasks * n_repeats

        # Use benchmark class name if desc not provided
        if self.desc is None:
            self.desc = f"Running {benchmark.__class__.__name__}"

        self._initialize_progress_bar()

    def on_task_repeat_end(self, benchmark: "Benchmark", report: Dict) -> None:
        """Called by benchmark framework when a task repeat completes."""
        with self._lock:
            self.current += 1
            status = report.get("status", "unknown")
            if status == "success":
                self.successful += 1

            # Update custom metrics from report
            metrics = self.update_metrics(report)
            if metrics:
                self._custom_metrics.update(metrics)

            self._update_progress(self.current, status if self.show_status else None)

    def on_run_end(self, benchmark: "Benchmark", results: List[Dict]) -> None:
        """Called by benchmark framework when run completes."""
        self._close_progress_bar()

    @abstractmethod
    def _initialize_progress_bar(self) -> None:
        """Initialize the progress bar display.

        Called once at the start of the benchmark run. Implementations should
        create and configure the underlying progress bar object.
        """
        pass

    @abstractmethod
    def _update_description(self, desc: str) -> None:
        """Update the progress bar description text.

        Args:
            desc: New description text to display
        """
        pass

    def set_metrics(self, **metrics: str) -> None:
        """Manually update custom metrics displayed in the progress bar.

        Call this method to set or update metrics at any time during benchmark execution.
        The progress bar will immediately reflect the changes.

        Args:
            **metrics: Key-value pairs to display (e.g., accuracy="95%", loss="0.23")

        Example:
            ```python
            progress_bar = TqdmProgressBarCallback()
            benchmark = MyBenchmark(callbacks=[progress_bar])

            # Update metrics during or after execution
            progress_bar.set_metrics(accuracy="95%", f1="0.87")
            progress_bar.set_metrics(avg_loss="0.23")  # Updates/adds metrics
            ```
        """
        with self._lock:
            self._custom_metrics.update(metrics)
            self._update_metrics()

    def update_metrics(self, report: Dict) -> Dict[str, str]:
        """Extract and return custom metrics from task execution reports.

        Override this method in a subclass to automatically display metrics extracted from
        benchmark task reports. Called by the framework after each task completes.

        The default implementation returns an empty dict (no automatic metrics).
        Use `set_metrics()` instead if you prefer manual metric updates.

        Args:
            report: Task execution report containing status, results, and evaluation data.
                Common keys include "status", "evaluation_result", "agent_response".

        Returns:
            Dictionary mapping metric names to string values for display.
            Return empty dict `{}` if no metrics should be added.

        Example:
            ```python
            class MyProgressBar(TqdmProgressBarCallback):
                def update_metrics(self, report):
                    # Extract metrics from evaluation results
                    if "evaluation_result" in report:
                        result = report["evaluation_result"]
                        return {
                            "accuracy": f"{result['accuracy']:.1%}",
                            "f1": f"{result['f1']:.2f}"
                        }
                    return {}  # No metrics for this report

            progress_bar = MyProgressBar()
            benchmark = MyBenchmark(callbacks=[progress_bar])
            benchmark.run(tasks)  # Metrics auto-update after each task
            ```
        """
        return {}

    @abstractmethod
    def _update_progress(self, current: int, status: Optional[str]) -> None:
        """Update the progress bar position and status.

        Called after each task repeat completes. Implementations should update
        the progress bar's position and display the current status.

        Args:
            current: Current iteration count (1-indexed)
            status: Execution status of the completed iteration ("success", "error", etc.),
                or None if show_status is False
        """
        pass

    @abstractmethod
    def _update_metrics(self) -> None:
        """Update custom metrics display.

        Called when set_metrics() is invoked. Implementations should refresh
        the progress bar to display updated metrics from self._custom_metrics.
        """
        pass

    @abstractmethod
    def _close_progress_bar(self) -> None:
        """Close and finalize the progress bar display.

        Called once at the end of the benchmark run. Implementations should
        clean up the progress bar object and release any resources.
        """
        pass


class TqdmProgressBarCallback(ProgressBarCallback):
    """Progress bar callback using tqdm (recommended default).

    Simple text-based progress bar that works in terminals and Jupyter notebooks.
    Displays task completion, success rate, and custom metrics.

    Example:
        ```python
        from maseval.core.callbacks.progress_bar import TqdmProgressBarCallback

        # Basic usage
        progress_bar = TqdmProgressBarCallback()
        benchmark = MyBenchmark(callbacks=[progress_bar])
        benchmark.run(tasks)

        # With custom description and metrics
        progress_bar = TqdmProgressBarCallback(desc="Evaluating agents")
        progress_bar.set_metrics(accuracy="95%", f1="0.87")
        benchmark.run(tasks)
        ```

    Args:
        desc: Custom description (defaults to "Running {BenchmarkClassName}")
        show_status: Show success counter (default: True)
        leave: Keep bar visible after completion (default: True)
        ncols: Width in characters (default: auto)
        bar_format: Custom tqdm format string (default: None)
    """

    def __init__(
        self,
        desc: Optional[str] = None,
        show_status: bool = True,
        leave: bool = True,
        ncols: Optional[int] = None,
        bar_format: Optional[str] = None,
    ):
        super().__init__(desc=desc, show_status=show_status)
        self.leave = leave
        self.ncols = ncols
        self.bar_format = bar_format
        self._pbar = None

    def _initialize_progress_bar(self) -> None:
        """Initialize tqdm progress bar. (Internal)"""
        from tqdm import tqdm

        self._pbar = tqdm(
            total=self.total,
            desc=self.desc,
            leave=self.leave,
            ncols=self.ncols,
            bar_format=self.bar_format,
            unit="task",
        )

    def _update_description(self, desc: str) -> None:
        """Update tqdm description. (Internal)"""
        if self._pbar is not None:
            self._pbar.set_description(desc)

    def _update_progress(self, current: int, status: Optional[str]) -> None:
        """Update tqdm progress. (Internal)"""
        if self._pbar is not None:
            # Build postfix from success counter and custom metrics
            postfix_parts = []

            if status is not None:
                postfix_parts.append(f"{self.successful}/{self.current} Successful")

            if self._custom_metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in self._custom_metrics.items())
                postfix_parts.append(metrics_str)

            if postfix_parts:
                self._pbar.set_postfix_str(" | ".join(postfix_parts))

            self._pbar.update(1)

    def _update_metrics(self) -> None:
        """Update custom metrics display. (Internal)"""
        if self._pbar is not None and self._custom_metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in self._custom_metrics.items())
            self._pbar.set_postfix_str(metrics_str)

    def _close_progress_bar(self) -> None:
        """Close tqdm progress bar. (Internal)"""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


class RichProgressBarCallback(ProgressBarCallback):
    """Progress bar callback using rich library.

    Visually enhanced progress bar with color formatting, rich text support,
    and improved aesthetics. Requires the `rich` library to be installed.

    Example:
        ```python
        from maseval.core.callbacks.progress_bar import RichProgressBarCallback

        # Basic usage
        progress_bar = RichProgressBarCallback()
        benchmark = MyBenchmark(callbacks=[progress_bar])
        benchmark.run(tasks)

        # With custom metrics
        progress_bar = RichProgressBarCallback(desc="Benchmarking")
        progress_bar.set_metrics(avg_score="0.89", correct="42/50")
        benchmark.run(tasks)
        ```

    Args:
        desc: Custom description (defaults to "Running {BenchmarkClassName}")
        show_status: Show colored success counter (default: True)
        transient: Remove bar after completion (default: False)
    """

    def __init__(
        self,
        desc: Optional[str] = None,
        show_status: bool = True,
        transient: bool = False,
    ):
        super().__init__(desc=desc, show_status=show_status)
        self.transient = transient
        self._progress = None
        self._task_id = None

    def _initialize_progress_bar(self) -> None:
        """Initialize rich progress bar. (Internal)"""
        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn

        # Build columns based on configuration
        columns = [
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ]

        # Add status and metrics columns if enabled
        if self.show_status or self._custom_metrics:
            columns.append(TextColumn("[bold]{task.fields[info]}"))

        self._progress = Progress(*columns, transient=self.transient)
        self._progress.start()
        self._task_id = self._progress.add_task(self.desc or "Running Benchmark", total=self.total, info="")

    def _update_description(self, desc: str) -> None:
        """Update rich progress bar description. (Internal)"""
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=desc)

    def _update_progress(self, current: int, status: Optional[str]) -> None:
        """Update rich progress bar. (Internal)"""
        if self._progress is not None and self._task_id is not None:
            # Build info text from success counter and custom metrics
            info_parts = []

            if status is not None:
                success_text = f"[green]{self.successful}[/green]/{self.current} Successful"
                info_parts.append(success_text)

            if self._custom_metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in self._custom_metrics.items())
                info_parts.append(metrics_str)

            info_text = " | ".join(info_parts) if info_parts else ""
            self._progress.update(self._task_id, advance=1, info=info_text)

    def _update_metrics(self) -> None:
        """Update custom metrics display. (Internal)"""
        if self._progress is not None and self._task_id is not None and self._custom_metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in self._custom_metrics.items())
            self._progress.update(self._task_id, info=metrics_str)

    def _close_progress_bar(self) -> None:
        """Close rich progress bar. (Internal)"""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
