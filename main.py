from src.api import app
import uvicorn
from src.settings import PORT, HOST


def main() -> None:
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
    )


if __name__ == "__main__":
    main()
