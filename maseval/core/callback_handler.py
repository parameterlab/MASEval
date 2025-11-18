from typing import Callable, List, Any


class CallbackHandler:
    def __init__(self):
        self.callbacks: List[Callable[..., None]] = []

    def register(self, callback: Callable[..., None]) -> None:
        """Register a new callback."""
        self.callbacks.append(callback)

    def deregister(self, callback: Callable[..., None]) -> None:
        """Remove an existing callback."""
        self.callbacks.remove(callback)

    def invoke(self, *args: Any, **kwargs: Any) -> None:
        """Invoke all registered callbacks."""
        for callback in self.callbacks:
            callback(*args, **kwargs)
