import fire
from src.indexer import CodeIndexer
from src.searcher import SearchSystem
from pathlib import Path
import json


class RagCLI:
    """Official Command-Line Interface for the RAG system."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Processes and indexes the vLLM repository.
        Args:
        max_chunk_size: Maximum character length for each chunk.
        """
        print(f"Starting indexing with max_chunk_size={max_chunk_size}...")
        # instancie indexeur pydantic
        indexer = CodeIndexer(max_chunk_size=max_chunk_size)
        # parcours du dossier brut vllm
        chunks = indexer.path_to_directory("data/raw/vllm-0.10.1")
        # sauvegarde dans "data/processed"
        indexer.save_index(chunks)

    def search(self, query: str, k: int = 10) -> None:
        """Searches for a single query in the indexed knowledge base.

        Args:
            query: The user's query string in natural language.
            k: The maximum number of relevant documents to return.
        """
        print(f"Searching for: '{query}' (top-{k})...")
        # instancie la classe de recherche
        searcher = SearchSystem()
        # effectue la recherche
        results = searcher.search(query=query, k=k)

        for i, src in enumerate(results, 1):
            print(f"\n[{i}] Result found:")
            print(f"  Path: {src.file_path}")
            print(f"  Pos: {src.first_character_index} ->"
                  f" {src.last_character_index}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results"
    ) -> None:
        """Processes a JSON dataset and saves the retrieval results."""
        searcher = SearchSystem()
        try:
            searcher.load_index_files()
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"Error: Failed to load index files. Aborting. Details: {e}")
            return

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                # On extrait la liste sous 'rag_questions' si le JSON est enveloppé dans un dict
                dataset = raw_data.get("rag_questions", raw_data) if isinstance(raw_data, dict) else raw_data
        except FileNotFoundError:
            print(f"Error: The dataset file '{dataset_path}' was not found.")
            return
        except json.JSONDecodeError:
            print(f"Error: The file '{dataset_path}' is corrupted or"
                  f" not a valid JSON.")
            return
        except (PermissionError, OSError) as e:
            print(f"Unexpected error while reading dataset: {e}")
            return

        search_results_list = []

        for i, item in enumerate(dataset):
            try:
                # Vérification que la clé 'query' existe
                if not isinstance(item, dict) or "question" not in item:
                    raise KeyError("Missing or invalid 'query' key in item.")

                query_text = item["question"]
                # Récupération sécurisée du question_id exigé par la moulinette
                q_id = item.get("question_id", f"q_{i}")

                sources = searcher.search(query=query_text, k=k)

                search_results_list.append({
                    "question_id": q_id,
                    "question": query_text,
                    "question_str": query_text,
                    "retrieved_sources": [src.model_dump() for src in sources]
                })

            except KeyError as ke:
                print(f"Warning [Item {i}]: {ke} Skipping line.")
            except (PermissionError, OSError) as e:
                print(f"Warning [Item {i}]: Unexpected error processing query."
                      f"Details: {e}. Skipping line.")

        try:
            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / "generated_search_results.json"

            final_output = {
                "search_results": search_results_list,
                "k": k
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)

            print(f"Dataset processed successfully! Results saved "
                  f"to {output_file}")

        except (PermissionError, FileNotFoundError) as e:
            print(f"Error: Failed to save results to disk. Details: {e}")


def main() -> None:
    """Main entry point for Python Fire."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()