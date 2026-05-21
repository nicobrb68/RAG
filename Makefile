# Variables
PYTHON = uv run python

.PHONY: install run debug clean lint lint-strict

install:
	@uv sync -q

run: install
	@$(PYTHON) main.py $(ARGS)

debug: install
	@uv run python -m pdb main.py

clean:
	@rm -rf .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint: install
	uv run flake8 src/ main.py
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src/ main.py

lint-strict: install
	uv run flake8 src/ main.py
	uv run mypy --strict src/ main.py