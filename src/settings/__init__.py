import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EYES_DIST = BASE_DIR / "../eyes/dist"
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "")
MODEL = os.environ.get("MODEL", "")
MAX_OUTPUT_CHARS = int(os.environ.get("MAX_OUTPUT_CHARS", "2000"))

assert PORT
assert HOST
assert MODEL
assert MAX_OUTPUT_CHARS
