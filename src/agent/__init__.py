from src.agent.base import get_base_agent
from src.agent.tools import get_tools


def get_agent(conversation_id: str):
    return get_base_agent(tools=get_tools(conversation_id=conversation_id))
