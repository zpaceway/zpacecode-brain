import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EYES_DIST_DIR = (
    Path(os.getenv("EYES_DIST_DIR", "")) if os.getenv("EYES_DIST_DIR") else None
)
PORT = int(os.environ.get("PORT", ""))
HOST = os.environ.get("HOST", "")
MODEL = os.environ.get("MODEL", "")
MAX_OUTPUT_CHARS = int(os.environ.get("MAX_OUTPUT_CHARS", "2000"))

assert PORT
assert HOST
assert MODEL
assert MAX_OUTPUT_CHARS
