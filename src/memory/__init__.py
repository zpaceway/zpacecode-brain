from fastapi import WebSocket


fetch_available_browsers: dict[str, WebSocket] = {}
fetch_responses: dict[str, str] = {}
untruncated_outputs: dict[str, str] = {}
