import subprocess
import asyncio
from src.utils import autolog, get_logger

logger = get_logger(__name__)


@autolog()
async def exec_bash_cmd(
    cmd: str,
) -> str:
    """Executes a bash command and returns the output or error message.

    Args:
        cmd (str): The bash command to execute.

    Returns:
        str: The output of stdout and stderr from the executed command, formatted as markdown.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ),
        )
        output = f"""## stdout
{result.stdout.decode()}

## stderr
{result.stderr.decode()}
"""
    except subprocess.CalledProcessError as e:
        output = f"""## stdout
{e.stdout.decode()}

## stderr
{e.stderr.decode()}
"""
    return output.strip()
