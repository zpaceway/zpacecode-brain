from typing import Callable
import dotenv
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from src.settings import MODEL

dotenv.load_dotenv()


def get_model():
    provider = MODEL.split("/")[0]
    version = MODEL.split("/")[1]
    match provider:
        case "anthropic":
            return Claude(version)
        case "openai":
            return OpenAIChat(version)
        case _:
            raise ValueError(f"Unsupported model provider: {provider}")


def get_agent(tools: list[Callable]) -> Agent:
    agent = Agent(
        name="Zpacecode Assistant",
        model=get_model(),
        system_message="""You are Zpacecode Assistant, an AI assistant designed to help users with a variety of tasks.

IMPORTANT: The only way to communicate with the user is by calling the send_message tool. Your returned text responses are NOT delivered to the user — they will never see them. If you want the user to know something, you must call send_message. Before making any tool call, always call send_message first to explain what you are about to do. After completing a task, call send_message to share the results. Never assume the user can see anything you haven't explicitly sent via send_message.""",
        tools=tools,
    )

    return agent
