from src.utils import get_logger
from src.agent.tools.exec_bash_cmd import exec_bash_cmd
from src.agent.tools.get_output_chunk import get_output_chunk
from src.agent.tools.get_fetch_tool import get_fetch_tool
from src.agent.tools.run_subagent import run_subagent

logger = get_logger(__name__)


def get_tools(conversation_id: str):
    return [
        exec_bash_cmd,
        get_output_chunk,
        get_fetch_tool(conversation_id),
        run_subagent,
    ]
