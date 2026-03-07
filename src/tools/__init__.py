import re
import subprocess
import threading
from fastapi import WebSocket
import datetime
import asyncio
import random
from src.memory import fetch_responses, fetch_available_browsers, untruncated_outputs
import lxml.html
from html_to_markdown import convert, ConversionOptions
from src.utils import autolog, get_logger
from src.utils import MAX_OUTPUT_CHARS

logger = get_logger(__name__)
_terminals: dict[str, subprocess.Popen] = {}
_terminal_locks: dict[str, threading.Lock] = {}

SENTINEL = "__CMD_DONE__"
CMD_TIMEOUT = 60


def _get_or_create_terminal(terminal_id: str) -> subprocess.Popen:
    if terminal_id in _terminals and _terminals[terminal_id].poll() is None:
        return _terminals[terminal_id]

    proc = subprocess.Popen(
        ["bash"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _terminals[terminal_id] = proc
    _terminal_locks[terminal_id] = threading.Lock()
    return proc


def _read_until_sentinel(proc: subprocess.Popen) -> str:
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        if SENTINEL in line:
            break
        lines.append(line)
    return "".join(lines)


@autolog()
async def exec_bash_cmd(
    cmd: str,
    terminal_id: str | None = None,
) -> str:
    """Executes a bash command and returns the output or error message.

    Args:
        cmd (str): The bash command to execute.
        terminal_id (str | None): The terminal ID to reuse a persistent shell session.
            If provided, the command runs in a long-lived bash process that preserves
            environment variables, working directory, and other state across calls.
            If None, the command runs in a one-shot subprocess.

    Returns:
        str: The output of stdout and stderr from the executed command, formatted as markdown.
    """
    if terminal_id is None:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
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

    proc = _get_or_create_terminal(terminal_id)
    lock = _terminal_locks[terminal_id]

    with lock:
        if proc.poll() is not None:
            proc = _get_or_create_terminal(terminal_id)

        assert proc.stdin is not None
        full_cmd = f"{cmd} 2>&1; echo {SENTINEL}\n"
        proc.stdin.write(full_cmd)
        proc.stdin.flush()

        result_lines: list[str] = []
        reader_done = threading.Event()

        def _read():
            result_lines.append(_read_until_sentinel(proc))
            reader_done.set()

        t = threading.Thread(target=_read, daemon=True)
        t.start()

        if not reader_done.wait(timeout=CMD_TIMEOUT):
            return "## Error\nCommand timed out."

        output = f"""## stdout and/or stderr
{"".join(result_lines)}
"""
        return output.strip()


@autolog(truncate_output=False)
async def get_partial_untruncated_output(
    output_id: str,
    offset: int,
    limit: int,
) -> str:
    """Retrieves a partial slice of the full untruncated output for a given output ID.
    Returns characters from offset to offset+limit. The limit can not exceed the max
    output chars limit imposed on any tool. You can call this tool multiple times with
    the same output ID and different offsets to page through the full output.

    Args:
        output_id (str): The ID of the truncated output to retrieve.
        offset (int): The character offset to start reading from.
        limit (int): The maximum number of characters to return.

    Returns:
        str: A partial slice of the untruncated output, or an error message if not found.
    """
    if limit > MAX_OUTPUT_CHARS:
        return f"## Error\nlimit cannot exceed {MAX_OUTPUT_CHARS} characters."

    full_output = untruncated_outputs.get(output_id)

    if full_output is None:
        return "## Error\nNo untruncated output found for the given ID."

    end = offset + limit

    return full_output[offset:end]


def get_send_message_tool(websocket: WebSocket, conversation_id: str):
    @autolog()
    async def send_message(message: str) -> str:
        """Send a message to the user.

        Args:
            message (str): The message to send.

        Returns:
            str: A confirmation message that the message was sent.
        """
        await websocket.send_json(
            {
                "type": "assistant",
                "conversation_id": conversation_id,
                "message": {
                    "created_at": (
                        datetime.datetime.now(datetime.timezone.utc).isoformat()
                    ),
                    "content": message,
                },
            }
        )
        return "Message sent to user."

    return send_message


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
