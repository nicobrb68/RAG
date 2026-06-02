

.PHONY: install run debug clean lint lint-strict command


MODULE_NAME = src


install:
	@echo "Installing dependencies with uv..."
	uv sync


run:
	@echo "Running the RAG CLI application..."
	uv run python -m $(MODULE_NAME) --help

command:
	uv run python -m $(MODULE_NAME)
   


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

# Configuration des chemins
STUDENT_PATH     := /home/nbarbosa/rag
MOULINETTE_PATH  := /home/nbarbosa/rag/moulinette_pkg/moulinette-ubuntu
EXAMS_DIR        := /home/nbarbosa/rag/exams_pkg/exams


SCRIPT_RETRIEVAL := $(EXAMS_DIR)/scripts/exam_retrieval.sh
SCRIPT_ANSWER    := $(EXAMS_DIR)/scripts/exam_answer.sh
SCRIPT_EDGES     := $(EXAMS_DIR)/scripts/exam_edge_cases.sh

.PHONY: eval-moulinette eval-search-docs eval-search-code eval-live-search eval-live-answer eval-quality-answers eval-edge-cases


eval-moulinette:
	@echo "=> [Q2] Lancement de la moulinette officielle en tâche de fond..."
	@"$(SCRIPT_RETRIEVAL)" --student-path "$(STUDENT_PATH)" --moulinette-path "$(MOULINETTE_PATH)"

eval-search-docs:
	@echo "=> [Q7a/b] Évaluation rapide du Recall (Docs Privés)..."
	$(eval LATEST_EVAL := $(shell ls -td /home/nbarbosa/rag/exams_pkg/evaluations/retrieval/*/ 2>/dev/null | head -n1))
	@uv run python -m src evaluate \
		"$(LATEST_EVAL)search_results/dataset_docs_private_results.json" \
		"/home/nbarbosa/rag/exams_pkg/data/datasets/private/AnsweredQuestions/dataset_docs_private.json"

eval-search-code:
	@echo "=> [Q7c/d] Évaluation rapide du Recall (Code Privé)..."
	$(eval LATEST_EVAL := $(shell ls -td /home/nbarbosa/rag/exams_pkg/evaluations/retrieval/*/ 2>/dev/null | head -n1))
	@uv run python -m src evaluate \
		"$(LATEST_EVAL)search_results/dataset_code_private_results.json" \
		"/home/nbarbosa/rag/exams_pkg/data/datasets/private/AnsweredQuestions/dataset_code_private.json"

eval-live-search:
	@echo "=> [Q4] Lancement de la recherche de démo en direct..."
	@uv run python -m src search "How to configure OpenAI server ?" -k 10


eval-live-answer:
	@echo "=> [Q5] Lancement de la génération de réponse en direct (Qwen)..."
	@uv run python -m src answer "How to configure OpenAI server ?" -k 10

eval-quality-answers:
	@echo "=> [Q8] Lancement du script de notation de la qualité des réponses..."
	@"$(SCRIPT_ANSWER)" --student-path "$(STUDENT_PATH)" --moulinette-path "$(MOULINETTE_PATH)"

eval-edge-cases:
	@echo "=> [Q11] Lancement du crash-test sur les 4 cas limites (Edge Cases)..."
	@"$(SCRIPT_EDGES)" --student-path "$(STUDENT_PATH)"