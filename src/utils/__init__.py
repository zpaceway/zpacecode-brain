import uuid
import logging
from functools import wraps
from typing import Any, Callable, Coroutine
from src.settings import MAX_OUTPUT_CHARS
from src.memory import untruncated_outputs


def get_logger(name: str):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


def truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text

    output_id = uuid.uuid4().hex
    untruncated_outputs[output_id] = text

    return f"""# Warning, truncated
Full output stored with ID: {output_id} with {len(text)} characters.
A max of {MAX_OUTPUT_CHARS} characters can be output for any tool.
Only the first {MAX_OUTPUT_CHARS} characters are shown here:

{text[:MAX_OUTPUT_CHARS]}"""


def autolog(truncate_output: bool = True):
    def wrapper(func: Callable[..., Coroutine[Any, Any, str]]):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> str:
            logger.info(
                f"Calling function: {func.__name__} with args: {args} kwargs: {kwargs}"
            )
            try:
                result = f"# Output\n{await func(*args, **kwargs)}"
            except Exception as e:
                result = (
                    truncate(f"# Output\n{str(e)}")
                    if truncate_output
                    else f"# Output\n{str(e)}"
                )

            if truncate_output:
                result = truncate(result)

            trimmed_result = (
                str(result)[:1000].replace("\n", "\\n").replace("\r", "\\r")
            )
            logger.info(f"Function {func.__name__} returned: {trimmed_result}")

            return result

        return wrapper

    return wrapper
