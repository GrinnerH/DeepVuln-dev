import functools
import logging
from typing import Any, Callable, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def log_io(func: Callable) -> Callable:
    """Log input/output for tool functions (copied from DeerFlow)."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        params = ", ".join(
            [*(str(arg) for arg in args), *(f"{k}={v}" for k, v in kwargs.items())]
        )
        logger.info("Tool %s called with parameters: %s", func_name, params)
        result = func(*args, **kwargs)
        logger.info("Tool %s returned: %s", func_name, result)
        return result

    return wrapper


class LoggedToolMixin:
    """Mixin to add logging to LangChain tool classes."""

    def _log_operation(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        tool_name = self.__class__.__name__.replace("Logged", "")
        params = ", ".join(
            [*(str(arg) for arg in args), *(f"{k}={v}" for k, v in kwargs.items())]
        )
        logger.debug("Tool %s.%s called with parameters: %s", tool_name, method_name, params)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        self._log_operation("_run", *args, **kwargs)
        result = super()._run(*args, **kwargs)
        logger.debug(
            "Tool %s returned: %s",
            self.__class__.__name__.replace("Logged", ""),
            result,
        )
        return result


def create_logged_tool(base_tool_class: Type[T]) -> Type[T]:
    """Factory to wrap a LangChain tool class with logging."""

    class LoggedTool(LoggedToolMixin, base_tool_class):
        pass

    LoggedTool.__name__ = f"Logged{base_tool_class.__name__}"
    return LoggedTool
