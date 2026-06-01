

.PHONY: install run debug clean lint lint-strict


MODULE_NAME = src


install:
	@echo "Installing dependencies with uv..."
	uv pip install -r pyproject.toml


run:
	@echo "Running the RAG CLI application..."
	uv run python -m $(MODULE_NAME) --help


debug:
	@echo "Launching RAG CLI in debug mode (pdb)..."
	uv run python -m pdb -m $(MODULE_NAME)


clean:
	@echo "Cleaning up caches and temporary files..."
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

lint:
	@echo "Running mandatory Flake8 and Mypy checks..."
	uv run flake8 $(MODULE_NAME)
	uv run mypy --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		$(MODULE_NAME)

lint-strict:
	@echo "Running strict quality checks..."
	uv run flake8 $(MODULE_NAME)
	uv run mypy $(MODULE_NAME)