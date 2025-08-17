.PHONY: help install-pixi setup-pixi install-dev install clean format check-pixi
.DEFAULT_GOAL := help

help:
	@echo "SmartSensor Development Setup"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check-pixi:
	@if command -v pixi >/dev/null 2>&1; then \
		echo "✓ Pixi is already installed"; \
		pixi --version; \
	else \
		echo "✗ Pixi is not installed"; \
		exit 1; \
	fi

install-pixi:
	@echo "Installing pixi..."
	@if command -v pixi >/dev/null 2>&1; then \
		echo "✓ Pixi is already installed"; \
		pixi --version; \
	else \
		echo "Installing pixi via curl..."; \
		curl -fsSL https://pixi.sh/install.sh | bash; \
		echo "✓ Pixi installed successfully"; \
		echo "Please restart your shell or run: source ~/.bashrc"; \
	fi

setup-pixi: install-pixi
	@echo "Setting up pixi environment..."
	@if [ ! -f "pixi.toml" ]; then \
		echo "Initializing pixi project..."; \
		pixi init --channel conda-forge; \
		echo "✓ Pixi project initialized"; \
	fi
	@echo "Adding Python 3.12..."
	@pixi add python=3.12
	@echo "Adding development dependencies..."
	@pixi add pip poetry
	@echo "✓ Pixi environment setup complete"

install-dev: setup-pixi 
	@echo "Installing package in development mode..."
	@pixi run pip install -e .
	@echo "✓ Package installed in development mode"

install: install-dev

install-poetry-deps:
	@echo "Installing Poetry dependencies..."
	@pixi run poetry install
	@echo "✓ Poetry dependencies installed"

shell:
	@echo "Starting pixi shell..."
	@pixi shell

run:
	@echo "Running smartsensor..."
	@pixi run smartsensor

format:
	@echo "Formatting code..."
	@pixi run python -m black smartsensor/

# PROJECT
ampicilline: 
	cd project/v1.1.0/Ampiciline_focal_1 && pixi run bash script.sh
	cd project/v1.1.0/Ampiciline_focal_2 && pixi run bash script.sh

clean:
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .pytest_cache/ .coverage
	@echo "✓ Cleanup complete"

info:
	@echo "Environment Information:"
	@echo "Pixi version:"
	@pixi --version 2>/dev/null || echo "Pixi not installed"
	@echo "Python version:"
	@pixi run python --version 2>/dev/null || echo "Python not available in pixi environment"
	@echo "Package info:"
	@pixi run pip show smartsensor 2>/dev/null || echo "Package not installed"

dev-setup: install-dev install-poetry-deps 
	@echo "✓ Development environment is ready!"
	@echo "Next steps:"
	@echo "  - Run 'make shell' to enter the environment"
	@echo "  - Run 'make run' to test the CLI"

update:
	@echo "Updating environment..."
	@pixi update
	@pixi run poetry update
	@echo "✓ Environment updated"
