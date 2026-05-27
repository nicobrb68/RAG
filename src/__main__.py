import json
from pathlib import Path
import fire
from src.indexer import CodeIndexer
from src.searcher import SearchSystem


class RagCLI:
    """Official Command-Line Interface for the RAG system."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Processes and indexes the vLLM repository.

        Args:
            max_chunk_size: Maximum character length for each chunk.
        """
        print(f"Starting indexing with max_chunk_size={max_chunk_size}...")
        indexer = CodeIndexer(max_chunk_size=max_chunk_size)
        chunks = indexer.path_to_directory("data/raw/vllm-0.10.1")
        indexer.save_index(chunks)

    def search(self, query: str, k: int = 10) -> None:
        """Searches for a single query in the indexed knowledge base.

        Args:
            query: The user's query string in natural language.
            k: The maximum number of relevant documents to return.
        """
        print(f"Searching for: '{query}' (top-{k})...")
        searcher = SearchSystem()
        results = searcher.search(query=query, k=k)

        for i, src in enumerate(results, 1):
            print(f"\n[{i}] Result found:")
            print(f"  Path: {src.file_path}")
            print(
                f"  Pos: {src.first_character_index} ->"
                f" {src.last_character_index}"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Processes a JSON dataset and saves the retrieval results."""
        searcher = SearchSystem()
        if "code" in dataset_path.lower():
            target_index = "code"
        else:
            target_index = "docs"

        try:
            searcher.load_index_files(index_type=target_index)
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"Error: Failed to load index files. Details: {e}")
            return

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                dataset = (
                    raw_data.get("rag_questions", raw_data)
                    if isinstance(raw_data, dict)
                    else raw_data
                )
        except FileNotFoundError:
            print(f"Error: The dataset file '{dataset_path}' was not found.")
            return
        except json.JSONDecodeError:
            print(f"Error: The file '{dataset_path}' is corrupted.")
            return
        except (PermissionError, OSError) as e:
            print(f"Unexpected error while reading dataset: {e}")
            return

        search_results_list = []

        for i, item in enumerate(dataset):
            try:
                if not isinstance(item, dict) or "question" not in item:
                    raise KeyError("Missing or invalid 'question' key.")

                query_text = item["question"]
                q_id = item.get("question_id", f"q_{i}")

                sources = searcher.search(query=query_text, k=k)

                search_results_list.append(
                    {
                        "question_id": q_id,
                        "question": query_text,
                        "question_str": query_text,
                        "retrieved_sources": [
                            src.model_dump() for src in sources
                        ],
                    }
                )

            except KeyError as ke:
                print(f"Warning [Item {i}]: {ke} Skipping line.")
            except (PermissionError, OSError) as e:
                print(
                    f"Warning [Item {i}]: Unexpected error. "
                    f"Details: {e}. Skipping line."
                )

        try:
            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)

            input_filename = Path(dataset_path).stem
            output_file = out_dir / f"{input_filename}_results.json"

            final_output = {"search_results": search_results_list, "k": k}

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)

            print(f"Dataset processed! Results saved to {output_file}")

        except (PermissionError, FileNotFoundError) as e:
            print(f"Error: Failed to save results to disk. Details: {e}")

    def answer(self, query: str, k: int = 5) -> None:
        """Answers a single query using the complete RAG pipeline."""
        from src.generator import AnswerGenerator

        searcher = SearchSystem()

        target_index = (
            "code"
            if any(
                x in query.lower()
                for x in [".py", "def ", "class ", "_"]
            )
            else "docs"
        )
        searcher.load_index_files(index_type=target_index)

        sources = searcher.search(query=query, k=k)

        contexts = []
        for src in sources:
            for chunk in searcher.all_chunks_raw:
                if (
                    chunk["source"]["file_path"] == src.file_path
                    and chunk["source"]["first_character_index"]
                    == src.first_character_index
                ):
                    contexts.append(chunk["text_content"])

        generator = AnswerGenerator(model_name="Qwen/Qwen3-0.6B")
        response = generator.generate_answer(question=query, contexts=contexts)

        print(response)

    def answer_dataset(
        self,
        dataset_path: str,
        k: int = 3,
        save_directory: str = "data/output/generation_results",
    ) -> None:
        """Processes a complete dataset to generate text answers."""
        from src.generator import AnswerGenerator
        from tqdm import tqdm

        searcher = SearchSystem()
        generator = AnswerGenerator()

        target_index = "code" if "code" in dataset_path.lower() else "docs"
        searcher.load_index_files(index_type=target_index)

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f).get("rag_questions", [])
        except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
            print(f"Error with dataset file : {e}")
            return

        generation_results_list = []

        # Boucle principale avec la barre de progression tqdm
        for item in tqdm(dataset, desc="Génération des réponses RAG"):
            query_text = item["question"]
            sources = searcher.search(query=query_text, k=k)

            contexts = [
                c["text_content"]
                for c in searcher.all_chunks_raw
                for src in sources
                if c["source"]["file_path"] == src.file_path
                and c["source"]["first_character_index"]
                == src.first_character_index
            ]

            answer_text = generator.generate_answer(
                question=query_text, contexts=contexts
            )

            # Nettoyage strict : uniquement les clés nécessaires
            generation_results_list.append(
                {
                    "question_id": item.get("question_id"),
                    "question": query_text,
                    "generated_answer": answer_text,
                }
            )

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / f"{Path(dataset_path).stem}_answers.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"generation_results": generation_results_list},
                    f,
                    indent=4,
                )
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"Error while saving result: {e}")

        print(f"\nResults saved to {output_file}")


def main() -> None:
    """Main entry point for Python Fire."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()