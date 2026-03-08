import asyncio
from textwrap import dedent
from src.agent import get_assistant_agent, get_planner_agent
from src.utils import get_logger
from src.settings import EYES_DIST_DIR
from agno.models.message import Message
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.memory import fetch_responses, fetch_available_browsers
from src.tools import (
    exec_bash_cmd,
    get_output_chunk,
    get_fetch_tool,
)
from agno.run.agent import RunOutput, ModelRequestCompletedEvent, RunContentEvent
import json


app = FastAPI()
logger = get_logger(__name__)


async def _handle_agent_run(
    websocket: WebSocket,
    messages: list[Message],
    conversation_id: str,
):
    try:
        logger.info(f"Starting agent run for conversation_id: {conversation_id}...")

        tools = [
            exec_bash_cmd,
            get_output_chunk,
            get_fetch_tool(conversation_id),
        ]
        planner_agent = get_planner_agent(tools=tools)
        logger.info(f"Running planner agent for conversation_id: {conversation_id}...")
        planner_output = await planner_agent.arun(
            [
                Message(
                    role="user",
                    content=dedent(
                        f"""Based on the following conversation between a user and and assistant, generate a plan for how the assistant should respond to the user's last message. The plan should be a step-by-step outline of how the assistant will arrive at the final response, including any tools it will use and in what order. Be as detailed as possible in the plan.
                
                        ## Conversation
                        {
                            "\n".join(
                                [
                                    f"* {message.role}: {message.content}"
                                    for message in messages
                                ]
                            )
                        }
                        """,
                    ),
                )
            ]
        )
        logger.info(
            f"Planner agent completed for conversation_id: {conversation_id}. Output: {planner_output.content}"
        )
        planner_output_message = Message(
            role="user",
            content=planner_output.content,
        )
        assistant_agent = get_assistant_agent(tools=tools)
        additional_assistant_messages = []
        reconstructed_assistant_message = ""

        async for event in assistant_agent.arun(
            messages + [planner_output_message],
            stream=True,
            stream_events=True,
            yield_run_output=True,
        ):
            if type(event) is ModelRequestCompletedEvent:
                if reconstructed_assistant_message:
                    additional_assistant_messages.append(
                        Message(
                            role="assistant",
                            content=reconstructed_assistant_message,
                        )
                    )

                reconstructed_assistant_message = ""
                await websocket.send_json(
                    {
                        "type": "history",
                        "conversation_id": conversation_id,
                        "messages": [
                            json.loads(message.model_dump_json())
                            for message in (messages + additional_assistant_messages)
                            if message.role != "system"
                            and message.id != planner_output_message.id
                        ],
                        "completed": False,
                    }
                )

            elif type(event) is RunContentEvent:
                reconstructed_assistant_message += event.content or ""

                if reconstructed_assistant_message.strip():
                    await websocket.send_json(
                        {
                            "type": "history",
                            "conversation_id": conversation_id,
                            "messages": [
                                json.loads(message.model_dump_json())
                                for message in (
                                    messages
                                    + additional_assistant_messages
                                    + [
                                        Message(
                                            role="assistant",
                                            content=reconstructed_assistant_message,
                                        )
                                    ]
                                )
                                if message.role != "system"
                                and message.id != planner_output_message.id
                            ],
                            "completed": False,
                        }
                    )

            elif type(event) is RunOutput:
                await websocket.send_json(
                    {
                        "type": "history",
                        "conversation_id": conversation_id,
                        "messages": [
                            json.loads(message.model_dump_json())
                            for message in (event.messages or [])
                            if message.role != "system"
                            and message.id != planner_output_message.id
                        ],
                        "completed": True,
                    }
                )
            else:
                logger.info(
                    f"Received event type from agent: {type(event)} for conversation_id: {conversation_id}"
                )

        logger.info(f"Agent run completed for conversation_id: {conversation_id}...")
    except Exception as e:
        await websocket.send_json(
            {
                "type": "error",
                "conversation_id": conversation_id,
                "error": str(e),
                "completed": True,
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
