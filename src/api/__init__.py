import asyncio
from src.agent import get_agent
from src.utils import get_logger
from src.settings import EYES_DIST_DIR
from agno.models.message import Message
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.memory import fetch_responses, fetch_available_browsers
from src.tools import (
    get_send_message_tool,
    exec_bash_cmd,
    get_partial_untruncated_output,
    get_fetch_tool,
)


app = FastAPI()
logger = get_logger(__name__)


async def _handle_agent_run(
    websocket: WebSocket,
    messages: list[Message],
    conversation_id: str,
):
    try:
        agent = get_agent(
            tools=[
                exec_bash_cmd,
                get_partial_untruncated_output,
                get_send_message_tool(websocket, conversation_id),
                get_fetch_tool(conversation_id),
            ]
        )

        result = await agent.arun(messages)

        await websocket.send_json(
            {
                "type": "history",
                "conversation_id": conversation_id,
                "messages": [
                    message.model_dump() for message in (result.messages or [])
                ],
            }
        )
    except Exception as e:
        await websocket.send_json(
            {
                "type": "error",
                "conversation_id": conversation_id,
                "error": str(e),
            }
        )


@app.websocket("/ws/agent/run/")
async def run(websocket: WebSocket):
    await websocket.accept()

    tasks: set[asyncio.Task] = set()

    try:
        while True:
            data = await websocket.receive_json()

            msg_type: str = data.get("type")
            conversation_id: str = data.get("conversation_id")

            if msg_type == "fetch":
                url: str = data.get("url")
                response: str = data.get("response")
                fetch_responses[f"{url}:{conversation_id}"] = response
                continue

            messages: list = data.get("messages", [])

            if not messages or not conversation_id:
                await websocket.send_json(
                    {
                        "type": "error",
                        "conversation_id": conversation_id,
                        "error": "Missing 'messages' or 'conversation_id' in the request.",
                    }
                )
                continue

            messages = [Message(**msg) for msg in messages]

            task = asyncio.create_task(
                _handle_agent_run(websocket, messages, conversation_id)
            )
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    except WebSocketDisconnect:
        for task in tasks:
            task.cancel()
        pass


@app.websocket("/ws/browser/register/")
async def register_browser(websocket: WebSocket):
    await websocket.accept()

    browser_id = str(id(websocket))
    fetch_available_browsers[browser_id] = websocket
    logger.info(f"Browser registered: {browser_id}")

    while True:
        try:
            data = await websocket.receive_json()

            logger.info(
                f"Received fetch response from browser {browser_id} for URL: {data.get('originalUrl')}"
            )
            original_url: str = data.get("originalUrl", "")
            html: str = data.get("html") or "<p>Empty HTML received</p>"
            conversation_id: str = data.get("conversation_id", "")

            if not original_url or not conversation_id:
                logger.warning(
                    f"Received invalid fetch response from browser {browser_id}. Missing 'originalUrl' or 'conversation_id'."
                )
                continue

            fetch_responses[f"{original_url}:{conversation_id}"] = html
        except WebSocketDisconnect:
            logger.info(f"Browser disconnected: {browser_id}")
            del fetch_available_browsers[browser_id]
            break


if EYES_DIST_DIR:
    app.mount(
        "/",
        StaticFiles(directory=EYES_DIST_DIR, html=True),
        name="static",
    )
