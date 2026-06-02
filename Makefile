

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

# Configuration des chemins
STUDENT_PATH     := /home/nbarbosa/rag
MOULINETTE_PATH  := /home/nbarbosa/rag/moulinette_pkg/moulinette-ubuntu
EXAMS_DIR        := /home/nbarbosa/rag/exams_pkg/exams

# Scripts officiels de l'examen
SCRIPT_RETRIEVAL := $(EXAMS_DIR)/scripts/exam_retrieval.sh
SCRIPT_ANSWER    := $(EXAMS_DIR)/scripts/exam_answer.sh
SCRIPT_EDGES     := $(EXAMS_DIR)/scripts/exam_edge_cases.sh

.PHONY: eval-moulinette eval-search-docs eval-search-code eval-live-search eval-live-answer eval-quality-answers eval-edge-cases

# 1. ETAPE PRELIMINAIRE : Lance la grosse moulinette (Linter + Recall en tâche de fond) [Q2]
eval-moulinette:
	@echo "=> [Q2] Lancement de la moulinette officielle en tâche de fond..."
	@"$(SCRIPT_RETRIEVAL)" --student-path "$(STUDENT_PATH)" --moulinette-path "$(MOULINETTE_PATH)"

# 2. EVAL SEARCH DIRECT (Docs) : Calcule instantanément ton Recall@5 Docs [Q7a/b]
eval-search-docs:
	@echo "=> [Q7a/b] Évaluation rapide du Recall (Docs Privés)..."
	$(eval LATEST_EVAL := $(shell ls -td /home/nbarbosa/rag/exams_pkg/evaluations/retrieval/*/ 2>/dev/null | head -n1))
	@uv run python -m src evaluate \
		"$(LATEST_EVAL)search_results/dataset_docs_private_results.json" \
		"/home/nbarbosa/rag/exams_pkg/data/datasets/private/AnsweredQuestions/dataset_docs_private.json"

# 3. EVAL SEARCH DIRECT (Code) : Calcule instantanément ton Recall@5 Code [Q7c/d]
eval-search-code:
	@echo "=> [Q7c/d] Évaluation rapide du Recall (Code Privé)..."
	$(eval LATEST_EVAL := $(shell ls -td /home/nbarbosa/rag/exams_pkg/evaluations/retrieval/*/ 2>/dev/null | head -n1))
	@uv run python -m src evaluate \
		"$(LATEST_EVAL)search_results/dataset_code_private_results.json" \
		"/home/nbarbosa/rag/exams_pkg/data/datasets/private/AnsweredQuestions/dataset_code_private.json"

# 4. DEMO LIVE : Test de recherche imposé par le barème [Q4]
eval-live-search:
	@echo "=> [Q4] Lancement de la recherche de démo en direct..."
	@uv run python -m src search "How to configure OpenAI server ?" -k 10

# 5. DEMO LIVE : Test de génération imposé par le barème [Q5]
eval-live-answer:
	@echo "=> [Q5] Lancement de la génération de réponse en direct (Qwen)..."
	@uv run python -m src answer "How to configure OpenAI server ?" -k 10

# 6. QUALITY EVAL : Lance le script d'évaluation de la qualité des réponses (3 questions) [Q8]
eval-quality-answers:
	@echo "=> [Q8] Lancement du script de notation de la qualité des réponses..."
	@"$(SCRIPT_ANSWER)" --student-path "$(STUDENT_PATH)" --moulinette-path "$(MOULINETTE_PATH)"

# 7. ROBUSTESSE : Lance le script de crash-test automatique sur les inputs dégénérés [Q11]
eval-edge-cases:
	@echo "=> [Q11] Lancement du crash-test sur les 4 cas limites (Edge Cases)..."
	@"$(SCRIPT_EDGES)" --student-path "$(STUDENT_PATH)"