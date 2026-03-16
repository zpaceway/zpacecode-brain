from src.memory import untruncated_outputs
from src.utils import autolog, get_logger
from src.utils import MAX_OUTPUT_CHARS

logger = get_logger(__name__)


@autolog(truncate_output=False)
async def get_output_chunk(
    output_id: str,
    offset: int,
    limit: int,
) -> str:
    """Retrieves a chunk of a previously truncated output by its ID.
    Returns characters from offset to offset+limit. The limit cannot exceed the max
    output chars limit imposed on any tool. Call this tool multiple times with different
    offsets to page through the full output.

    Args:
        output_id (str): The ID of the output to retrieve a chunk from.
        offset (int): The character offset to start reading from.
        limit (int): The maximum number of characters to return.

    Returns:
        str: The requested output chunk, or an error message if not found.
    """
    if limit > MAX_OUTPUT_CHARS:
        return f"## Error\nlimit cannot exceed {MAX_OUTPUT_CHARS} characters."

    full_output = untruncated_outputs.get(output_id)

    if full_output is None:
        return "## Error\nNo untruncated output found for the given ID."

    end = offset + limit

    return full_output[offset:end]
