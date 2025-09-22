.PHONY: install setup ingest run build run-docker stop logs clean

APP_NAME = guardiao-dos-direitos
DOCKER_IMAGE = $(APP_NAME)-app
DOCKER_CONTAINER = $(APP_NAME)-container
PORT = 8501

install:
	pip install --upgrade pip
	pip install -r requirements.txt

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

ingest:
	python -m core.ingestion

run:
	streamlit run app/main.py

clean:
	rm -rf .venv __pycache__ */__pycache__ .pytest_cache
	find . -type f -name '*.py[co]' -delete

build:
	docker build -t $(DOCKER_IMAGE) .

run-docker:
	docker run -p $(PORT):$(PORT) --rm --name $(DOCKER_CONTAINER) \
		-v $(shell pwd):/app $(DOCKER_IMAGE)

stop:
	docker stop $(DOCKER_CONTAINER) || true

logs:
	docker logs -f $(DOCKER_CONTAINER)

evaluate:
	python -m eval.evaluate