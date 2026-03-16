import json
import os
from agno.models.message import Message
from fastapi import WebSocket
from src.settings import MESSAGES_DIR


fetch_available_browsers: dict[str, WebSocket] = {}
fetch_responses: dict[str, str] = {}
untruncated_outputs: dict[str, str] = {}


class ConversationsMemory:
    conversation_id_messages_mapping: dict[str, list[Message]] = {}

    def get_conversation_ids(self) -> list[str]:
        return list(self.conversation_id_messages_mapping.keys())

    def get_messages(self, conversation_id: str) -> list[Message]:
        return self.conversation_id_messages_mapping.get(conversation_id, [])

    def set_messages(self, conversation_id: str, messages: list[Message]):
        with open(f"{MESSAGES_DIR}/{conversation_id}.json", "w") as f:
            json.dump([message.model_dump() for message in messages], f)
        self.conversation_id_messages_mapping[conversation_id] = messages

    def load_messages(self):
        for filename in os.listdir(MESSAGES_DIR):
            if not filename.endswith(".json"):
                continue

            conversation_id = filename[:-5]
            with open(f"{MESSAGES_DIR}/{filename}", "r") as f:
                messages_data = json.load(f)
                messages = [
                    Message.model_validate(message) for message in messages_data
                ]
                self.conversation_id_messages_mapping[conversation_id] = messages


conversations_memory = ConversationsMemory()
conversations_memory.load_messages()
