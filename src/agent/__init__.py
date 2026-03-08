import json
from typing import Callable
from agno.agent import Agent, Message
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from fastapi import WebSocket
from src.settings import MODEL
from src.utils import get_logger
from agno.run.agent import RunOutput


logger = get_logger(__name__)


def get_model():
    provider = MODEL.split("/")[0]
    version = MODEL.split("/")[1]
    match provider:
        case "anthropic":
            return Claude(version)
        case "openai":
            return OpenAIChat(version)
        case "ollama":
            return Ollama(version)
        case _:
            raise ValueError(f"Unsupported model provider: {provider}")


def get_post_hook(
    conversation_id: str,
    prev_messages: list[Message],
    websocket: WebSocket,
):

    async def get_post_hook(*args, **kwargs) -> None:
        run_output: RunOutput = kwargs.get("run_output")  # type: ignore

        if run_output is None:
            return

        messages = {
            "type": "history",
            "conversation_id": conversation_id,
            "messages": [
                message.model_dump(mode="json", exclude={"metrics"})
                for message in (prev_messages + (run_output.messages or []))
                if message.role != "system"
            ],
            "completed": False,
        }

        await websocket.send_json(messages)

    return get_post_hook


def get_agent(
    websocket: WebSocket,
    conversation_id: str,
    prev_messages: list[Message],
    tools: list[Callable],
) -> Agent:
    agent = Agent(
        name="Zpacecode Assistant",
        model=get_model(),
        system_message="""You are Zpacecode Assistant, an AI assistant designed to help users with a variety of tasks.""",
        tools=tools,
        pre_hooks=[],
        post_hooks=[get_post_hook(conversation_id, prev_messages, websocket)],
        tool_hooks=[],
        reasoning=True,
    )

    return agent
