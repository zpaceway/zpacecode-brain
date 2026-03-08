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
                        f"""# Execution Planning Prompt

                        ## Task

                        Based on the following conversation between a user and an assistant,
                        generate a **detailed execution plan** describing how the assistant
                        should respond to the **user's last message**.

                        The plan must be a **step-by-step outline** explaining how the assistant
                        will arrive at the final response. It should clearly describe:

                        -   the reasoning process
                        -   the decomposition of the task into smaller subtasks
                        -   any tools that should be used
                        -   the order in which tools should be used
                        -   which steps depend on previous steps
                        -   which steps can be **executed in parallel**

                        The plan should be **implementation-oriented**, focusing on how an AI
                        system would reliably execute the task in a production environment.

                        ------------------------------------------------------------------------

                        # Execution Constraints and Best Practices

                        The plan must explicitly incorporate the following principles.

                        ------------------------------------------------------------------------

                        # 1. Context Window Management

                        Be extremely careful with **context size limitations**. The plan should
                        minimize unnecessary context growth.

                        Strategies should include:

                        -   Avoid repeatedly injecting the entire conversation or large
                            documents into prompts.
                        -   Use **summaries instead of raw content** when previous outputs
                            become large.
                        -   Maintain **compressed representations** of earlier results when
                            possible.
                        -   Pass only the **minimal required context** for each step.
                        -   Avoid concatenating multiple large intermediate artifacts into a
                            single prompt.
                        -   Prefer **retrieval-based access** to intermediate results instead of
                            storing everything in the context window.
                        -   Process large documents in **smaller chunks** rather than loading
                            them entirely.
                        -   Maintain **rolling summaries** of long outputs to reduce token
                            usage.
                        -   Track a **token budget** per step to ensure the system does not
                            exceed context limits.
                        -   If previous outputs grow too large, store them externally and
                            reference them instead of embedding them in prompts.

                        The plan should explicitly note when **context compression,
                        summarization, or retrieval strategies** should occur.

                        ------------------------------------------------------------------------

                        # 2. Avoid Generating Large Outputs in a Single Step

                        The assistant should **avoid generating very large outputs in a single
                        pass**.

                        If the requested output could be large (for example a long report,
                        codebase, dataset, analysis, or document), the plan should recommend:

                        -   Breaking the output into **sections or modules**
                        -   Generating each section independently
                        -   Saving intermediate results
                        -   Combining results later in a final aggregation step

                        Large outputs should **never be generated monolithically** when they can
                        be modularized.

                        ------------------------------------------------------------------------

                        # 3. Intermediate Result Storage

                        If intermediate outputs are large or important, the plan should
                        recommend **saving them to disk** instead of keeping them in the model
                        context.

                        Example strategy:

                            /tmp/task_run/
                                section_1.txt
                                section_2.txt
                                section_3.txt
                                section_summary_1.txt
                                section_summary_2.txt
                                metadata.json

                        Possible storage strategies:

                        -   Save generated sections
                        -   Save summaries of each section
                        -   Save metadata describing generated artifacts
                        -   Save intermediate analysis results

                        Benefits include:

                        -   reducing context pressure
                        -   enabling later retrieval
                        -   enabling modular assembly of the final result
                        -   improving reproducibility

                        The plan should specify **what to store, when to store it, and when to
                        reload it**.

                        ------------------------------------------------------------------------

                        # 4. Modular Task Decomposition

                        The plan should decompose the task into **small, well-defined modules**.

                        Each module should:

                        -   have a clear input
                        -   have a clear output
                        -   avoid unnecessary dependencies
                        -   be independently executable when possible
                        -   be reusable

                        Examples of modules:

                        -   input parsing
                        -   information retrieval
                        -   preprocessing
                        -   analysis
                        -   summarization
                        -   transformation
                        -   validation
                        -   formatting
                        -   aggregation

                        Modularity improves:

                        -   reliability
                        -   debuggability
                        -   scalability
                        -   context efficiency

                        ------------------------------------------------------------------------

                        # 5. Parallelization Opportunities

                        The plan should identify **independent steps that can run in parallel**.

                        Examples include:

                        -   processing multiple documents simultaneously
                        -   generating independent sections of a report
                        -   analyzing multiple datasets
                        -   performing retrieval queries concurrently
                        -   summarizing independent sections

                        Parallel execution reduces latency and improves scalability.

                        The plan should clearly mark steps such as:

                            Step 4A - Run in parallel
                            Step 4B - Run in parallel
                            Step 4C - Run in parallel

                        ------------------------------------------------------------------------

                        # 6. Progressive Aggregation

                        When outputs from multiple modules must be combined, the plan should
                        recommend **progressive aggregation**.

                        Instead of merging everything at once:

                        1.  Generate modular outputs
                        2.  Summarize each module
                        3.  Combine summaries
                        4.  Generate a final synthesized output

                        This prevents large context overload.

                        ------------------------------------------------------------------------

                        # 7. Intermediate Summarization

                        If intermediate outputs are long, the plan should recommend generating
                        **compact summaries**.

                        Example:

                            section_output.txt
                            section_summary.txt

                        Only the **summary** should be passed forward unless the full output is
                        required.

                        ------------------------------------------------------------------------

                        # 8. Checkpointing

                        For long or complex workflows, the plan should include **checkpointing
                        strategies**.

                        Examples:

                            checkpoint_1_complete
                            checkpoint_2_complete
                            checkpoint_3_complete

                        Checkpointing helps:

                        -   resume partially completed workflows
                        -   prevent recomputation
                        -   improve reliability

                        ------------------------------------------------------------------------

                        # 9. Error Resilience

                        The plan should include safeguards such as:

                        -   validating outputs before using them
                        -   detecting empty or malformed results
                        -   retrying failed steps
                        -   fallback strategies when tools fail
                        -   verifying assumptions before proceeding to later steps

                        ------------------------------------------------------------------------

                        # 10. Final Response Assembly

                        The final step should describe how to assemble the final answer:

                        -   retrieve stored intermediate outputs
                        -   combine summaries or section outputs
                        -   ensure coherence and logical ordering
                        -   format the final response
                        -   verify correctness and completeness

                        If the final result is large, prefer:

                        -   a **summarized response**
                        -   references to generated sections
                        -   or modular outputs instead of injecting everything into the final
                            message.

                        ------------------------------------------------------------------------

                        # Output Format

                        The response should be a **clear numbered step-by-step execution plan**.

                        Each step should include:

                        -   step description
                        -   inputs
                        -   outputs
                        -   tool usage (if applicable)
                        -   notes about context management
                        -   notes about storage
                        -   notes about parallelization

                        Example structure:

                            Step 1 - Parse the user's last request
                            Step 2 - Identify required subtasks
                            Step 3 - Retrieve necessary information
                            Step 4 - Parallel processing of subtasks
                            Step 5 - Store intermediate outputs
                            Step 6 - Generate summaries
                            Step 7 - Aggregate results
                            Step 8 - Generate final response

                        Each step should contain **implementation notes describing how the
                        system should execute the step efficiently and safely.**

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
