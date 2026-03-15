import re
import subprocess
import asyncio
import random
from src.agent import get_agent
from src.memory import fetch_responses, fetch_available_browsers, untruncated_outputs
import lxml.html
from html_to_markdown import convert, ConversionOptions
from src.utils import autolog, get_logger
from src.utils import MAX_OUTPUT_CHARS
from agno.models.message import Message

logger = get_logger(__name__)


def get_tools(conversation_id: str):
    return [
        exec_bash_cmd,
        get_output_chunk,
        get_fetch_tool(conversation_id),
        get_run_subagent_tool(conversation_id),
    ]


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


def get_fetch_tool(conversation_id: str):
    @autolog()
    async def fetch(
        url: str,
    ) -> str:
        """Fetches a URL and returns the content as markdown.

        Args:
            url (str): The URL to fetch.

        Returns:
            str: The content of the URL as markdown, or an error message if fetching fails.
        """

        while not fetch_available_browsers:
            logger.info("No available browsers to fetch the URL. Waiting...")
            await asyncio.sleep(1)

        random_browser_key = random.choice(list(fetch_available_browsers.keys()))
        browser_ws = fetch_available_browsers[random_browser_key]
        await browser_ws.send_json(
            {
                "type": "fetch",
                "conversation_id": conversation_id,
                "url": url,
            }
        )

        elapsed = 0

        while True:
            await asyncio.sleep(1)
            elapsed += 1

            if elapsed > 120:
                return "## Error\nFetching the URL timed out."

            raw_html = fetch_responses.get(f"{url}:{conversation_id}")

            if not raw_html:
                logger.info(f"Waiting for browser to fetch the URL: {url}")
                continue

            del fetch_responses[f"{url}:{conversation_id}"]
            html = lxml.html.fromstring(raw_html)
            tags_to_remove = [
                # Scripts and styles
                "script",
                "style",
                "noscript",
                "template",
                # Media
                "img",
                "picture",
                "source",
                "video",
                "audio",
                "track",
                "canvas",
                "svg",
                # Embeds
                "iframe",
                "embed",
                "object",
                "applet",
                # Metadata
                "meta",
                "link",
                "base",
                # Misc
                "map",
                "area",
                "param",
                "portal",
                "slot",
            ]

            for tag in tags_to_remove:
                for element in html.findall(f".//{tag}"):
                    element.drop_tree()  # type: ignore

            attrs_to_remove = [
                "class",
                "id",
                "style",
                "onclick",
                "onmouseover",
                "onerror",
                "data-*",
            ]

            for attr in attrs_to_remove:
                for element in html.findall(f".//*[@{attr}]"):
                    del element.attrib[attr]

            markdown = convert(
                lxml.html.tostring(html, encoding="unicode"),
                options=ConversionOptions(
                    heading_style="atx",
                    list_indent_width=2,
                ),
            )
            markdown = re.sub(r"\n{3,}", "\n\n", markdown)
            markdown = "\n".join(line.rstrip() for line in markdown.splitlines())

            return markdown.strip()

    return fetch


def get_run_subagent_tool(conversation_id: str):
    @autolog(truncate_output=False)
    async def run_subagent(message: str) -> str:
        """Runs a sub-agent with the provided message as input. The sub-agent will have access to the same tools as the main agent, but will operate with a fresh context. This is useful for delegating complex subtasks that require multiple steps or tool calls, allowing the main agent to offload work while maintaining modularity and context efficiency.

        Args:
            message (str): The input message to provide to the sub-agent. This should include all necessary information and instructions for the sub-agent to complete its task.

        Returns:
            str: The final response generated by the sub-agent after processing the input message and utilizing any necessary tools.
        """
        subagent = get_agent(
            tools=get_tools(conversation_id=conversation_id),
        )
        logger.info(f"Running sub-agent for conversation_id: {conversation_id}...")
        subagent_output = await subagent.arun(
            [
                Message(
                    role="user",
                    content=message,
                )
            ]
        )

        logger.info(
            f"Sub-agent completed for conversation_id: {conversation_id}. Output: {subagent_output.content}"
        )
        content: str = subagent_output.content  # type: ignore

        return content

    return run_subagent
