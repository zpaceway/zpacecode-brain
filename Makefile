-include .env
export

PYTHON ?= ./.venv/bin/python
NGROK ?= ngrok

run:
	${PYTHON} -m uvicorn main:app --host ${HOST} --port ${PORT}

expose:
	${NGROK} http ${PORT} --domain ${NGROK_DOMAIN}
