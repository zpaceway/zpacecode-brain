import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EYES_DIST_DIR = (
    Path(os.getenv("EYES_DIST_DIR", "")) if os.getenv("EYES_DIST_DIR") else None
)
MESSAGES_DIR = BASE_DIR / ".messages"
PORT = int(os.environ.get("PORT", ""))
HOST = os.environ.get("HOST", "")
MODEL = os.environ.get("MODEL", "")
MAX_OUTPUT_CHARS = int(os.environ.get("MAX_OUTPUT_CHARS", "2000"))
APP_TOKEN = os.environ.get("APP_TOKEN", "")

MESSAGES_DIR.mkdir(exist_ok=True)
assert PORT
assert HOST
assert MODEL
assert MAX_OUTPUT_CHARS
assert APP_TOKEN
