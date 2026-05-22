# ==============================================================================
# CONFIGURATION ET CHEMINS
# ==============================================================================
MOULINETTE = ./moulinette/moulinette-ubuntu
RESULT_JSON = data/output/search_results/generated_search_results.json

DS_DOCS_UNANSWERED = datasets_public/public/UnansweredQuestions/dataset_docs_public.json
DS_DOCS_ANSWERED   = datasets_public/public/AnsweredQuestions/dataset_docs_public.json

DS_CODE_UNANSWERED = datasets_public/public/UnansweredQuestions/dataset_code_public.json
DS_CODE_ANSWERED   = datasets_public/public/AnsweredQuestions/dataset_code_public.json

# Paramètres imposés par la moulinette
K = 5
MAX_LENGTH = 2000
THRESHOLD = 0.5

# Couleurs pour l'affichage de la console
GREEN  = \033[1;32m
YELLOW = \033[1;33m
RED    = \033[1;31m
BLUE   = \033[1;34m
RESET  = \033[0m

.PHONY: all clean index test-docs test-code test-all help

# ==============================================================================
# COMMANDES PRINCIPALES
# ==============================================================================

help:
	@echo "$(BLUE)=== RAG AUTOMATION SYSTEM ===$(RESET)"
	@echo "Commandes disponibles :"
	@echo "  make index      - Supprime les anciens index et reconstruit la base BM25"
	@echo "  make test-docs  - Cherche et évalue le dataset de DOCUMENTATION"
	@echo "  make test-code  - Cherche et évalue le dataset de CODE SOURCE"
	@echo "  make test-all   - Exécute l'indexation complète et lance les deux tests à la suite"
	@echo "  make clean      - Nettoie les fichiers d'indexation et les résultats générés"

clean:
	@echo "$(YELLOW) Nettoyage des anciennes données d'index et résultats...$(RESET)"
	rm -rf data/processed/bm25_index
	rm -rf data/processed/chunks
	rm -f $(RESULT_JSON)
	@echo "$(GREEN) Nettoyage terminé !$(RESET)"

index: clean
	@echo "$(BLUE) Extraction et indexation du jeu de données brut...$(RESET)"
	uv run python main.py index

test-docs:
	@echo "\n$(BLUE)============== TEST DATASET : DOCUMENTATION ==============$(RESET)"
	@echo "$(YELLOW)1. Exécution de la recherche sur les questions Docs...$(RESET)"
	uv run python main.py search_dataset $(DS_DOCS_UNANSWERED) --k $(K)
	@echo "$(YELLOW)2. Lancement de l'évaluation par la moulinette...$(RESET)"
	$(MOULINETTE) evaluate_student_search_results \
		$(RESULT_JSON) \
		$(DS_DOCS_ANSWERED) \
		--k $(K) \
		--max_context_length $(MAX_LENGTH) \
		--threshold $(THRESHOLD)

test-code:
	@echo "\n$(BLUE)============== TEST DATASET : CODE SOURCE ==============$(RESET)"
	@echo "$(YELLOW)1. Exécution de la recherche sur les questions Code...$(RESET)"
	uv run python main.py search_dataset $(DS_CODE_UNANSWERED) --k $(K)
	@echo "$(YELLOW)2. Lancement de l'évaluation par la moulinette...$(RESET)"
	$(MOULINETTE) evaluate_student_search_results \
		$(RESULT_JSON) \
		$(DS_CODE_ANSWERED) \
		--k $(K) \
		--max_context_length $(MAX_LENGTH) \
		--threshold $(THRESHOLD)

test-all: index test-docs test-code
	@echo "\n$(GREEN) Pipeline complet exécuté avec succès !Vérifie tes scores ci-dessus.$(RESET)"