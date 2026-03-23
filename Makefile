.PHONY: test test-integration lint format type-check docs clean install

test:
	pytest tests/unit

test-integration:
	pytest tests/integration --integration

lint:
	ruff check . && ruff format --check .

format:
	ruff format .

type-check:
	mypy relay

docs:
	pdoc3 --html relay -o docs

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name dist -exec rm -rf {} +
	find . -type d -name build -exec rm -rf {} +
	find . -name '*.pyc' -delete

install:
	pip install -e '.[dev,all]'
