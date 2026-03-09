-include .env
export

PYTHON ?= ./.venv/bin/python
NGROK ?= ngrok
UV ?= uv

sync:
	${UV} sync --dev

dev:
	${PYTHON} -m uvicorn main:app --reload --host ${HOST} --port ${PORT}

run:
	${PYTHON} -m uvicorn main:app --host ${HOST} --port ${PORT}

expose:
	${NGROK} http ${PORT} --domain ${NGROK_DOMAIN}
