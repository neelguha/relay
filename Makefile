.PHONY: install install-dev build-deps build clean test publish publish-test help

help:
	@echo "Available commands:"
	@echo "  make install      - Install the package"
	@echo "  make install-dev  - Install the package with dev dependencies"
	@echo "  make build-deps   - Install build dependencies (build, twine)"
	@echo "  make build        - Build the package"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make test         - Run tests"
	@echo "  make publish      - Publish to PyPI (requires credentials)"
	@echo "  make publish-test - Publish to TestPyPI (requires credentials)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

build-deps:
	@echo "Installing build dependencies..."
	@pip install build twine || (echo "Error: Could not install build dependencies. Make sure you have network access and pip is working." && exit 1)

build:
	@echo "Checking for build tool..."
	@python -c "import build" 2>/dev/null || (echo "Error: 'build' package not found. Run 'make build-deps' or 'pip install build twine' first." && exit 1)
	@echo "Building package..."
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

test:
	pytest

publish: build
	@echo "Publishing to PyPI..."
	@python -c "import twine" 2>/dev/null || (echo "Error: 'twine' package not found. Run 'make build-deps' or 'pip install twine' first." && exit 1)
	@if [ -f ~/.pypirc ]; then \
		echo "✓ Using credentials from ~/.pypirc"; \
		twine upload dist/*; \
	elif [ -f .pypirc ]; then \
		echo "⚠ Found .pypirc in project directory. Twine reads from ~/.pypirc by default."; \
		echo "  Copy it to ~/.pypirc or use: TWINE_USERNAME and TWINE_PASSWORD environment variables"; \
		twine upload dist/*; \
	else \
		echo "⚠ No ~/.pypirc file found. Twine will prompt for credentials."; \
		echo "  Create ~/.pypirc with your PyPI token, or use TWINE_USERNAME/TWINE_PASSWORD env vars."; \
		twine upload dist/*; \
	fi

publish-test: build
	@echo "Publishing to TestPyPI..."
	@python -c "import twine" 2>/dev/null || (echo "Error: 'twine' package not found. Run 'make build-deps' or 'pip install twine' first." && exit 1)
	@if [ -f ~/.pypirc ]; then \
		echo "✓ Using credentials from ~/.pypirc"; \
		twine upload --repository testpypi dist/*; \
	elif [ -f .pypirc ]; then \
		echo "⚠ Found .pypirc in project directory. Twine reads from ~/.pypirc by default."; \
		echo "  Copy it to ~/.pypirc or use: TWINE_USERNAME and TWINE_PASSWORD environment variables"; \
		twine upload --repository testpypi dist/*; \
	else \
		echo "⚠ No ~/.pypirc file found. Twine will prompt for credentials."; \
		echo "  Create ~/.pypirc with your PyPI token, or use TWINE_USERNAME/TWINE_PASSWORD env vars."; \
		twine upload --repository testpypi dist/*; \
	fi
