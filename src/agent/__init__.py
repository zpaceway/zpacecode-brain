from textwrap import dedent
from typing import Callable
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from src.settings import MODEL
from src.utils import get_logger


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


def get_assistant_agent(
    tools: list[Callable],
) -> Agent:
    agent = Agent(
        name="Zpacecode Assistant",
        model=get_model(),
        system_message="""You are Zpacecode Assistant, an AI assistant designed to help users with a variety of tasks.""",
        tools=tools,
    )

    return agent


def get_planner_agent(tools: list[Callable]) -> Agent:
    agent = Agent(
        name="Zpacecode Planner",
        model=get_model(),
        system_message=dedent("""You are Zpacecode Planner, an AI assistant designed to help agents plan how to answer user queries effectively.
        
        Don't actually execute any of the tools yourself, just outline a plan for how an assistant should use the tools to arrive at a final answer for the user query. Be as detailed as possible in the plan."""),
        tools=tools,
    )

    return agent
