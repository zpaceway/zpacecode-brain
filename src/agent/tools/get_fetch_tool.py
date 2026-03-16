import re
import asyncio
import random
from src.memory import fetch_responses, fetch_available_browsers
import lxml.html
from html_to_markdown import convert, ConversionOptions
from src.utils import autolog, get_logger

logger = get_logger(__name__)


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
