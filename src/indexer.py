import sys
import json
from pathlib import Path
from typing import List
import bm25s
from pydantic import BaseModel
from src.models import MinimalSource, ChunkStorage


class CodeIndexer(BaseModel):
    """Handles the configuration and execution of database chunking.

    This class complies with Pydantic validation rules requested
    by the project guidelines.
    """

    max_chunk_size: int = 2000

    def _chunk_data(self, full_data: str) -> List[str]:
        """Slices raw string text into configurable length pieces."""
        chunked_data: List[str] = []
        start = 0
        while start < len(full_data):
            end = start + self.max_chunk_size
            chunk = full_data[start:end]
            chunked_data.append(chunk)
            start = end
        return chunked_data

    def read_then_chunk(self, path: Path) -> List[ChunkStorage]:
        """Reads a specific file path and creates localized Pydantic chunks."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                full_data = f.read()
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error with the data file: {e}, End of program")
            sys.exit(1)

        chunked_data = self._chunk_data(full_data)
        files_sources = []

        for i, chunk in enumerate(chunked_data):
            start = i * self.max_chunk_size
            end = start + len(chunk)
            source = MinimalSource(
                file_path=str(path),
                first_character_index=start,
                last_character_index=end
            )
            full_info = ChunkStorage(
                source=source,
                text_content=chunk
            )
            files_sources.append(full_info)

        print(f"Successfully created {len(chunked_data)}"
              f" chunks for {path.name}")
        return files_sources

    def path_to_directory(self, path_dir: str) -> List[ChunkStorage]:
        """Scans a repository to process every valid Python and
        Markdown file."""
        main_directory = Path(path_dir)
        all_sources = []

        for file_path in main_directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".py", ".md"]:
                print(f"File found : {file_path}")
                source_files = self.read_then_chunk(file_path)
                all_sources.extend(source_files)

        print(f"End of indexation. Total chunks created: {len(all_sources)}")
        return all_sources

    def save_index(self, all_chunk: List[ChunkStorage],
                   destination_path: str = "data/processed") -> None:
        """Stores BM25 statistics and raw chunk metadata onto the disk."""
        # objet path qui permet de cree de nouveau sous dossier grace a path
        directory_path = Path(destination_path)
        directory_bm25 = directory_path / "bm25_index"
        directory_chunks = directory_path / "chunks"

        try:
            # on cree les dossiers
            directory_bm25.mkdir(parents=True, exist_ok=True)
            directory_chunks.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            print(f"Error: Cannot create storage directory: ({e})")
            sys.exit(1)
        # faut donner le texte a bm25
        texte_only: List[str] = [chunk.text_content for chunk in all_chunk]

        try:
            # init de la classe
            index_bm25 = bm25s.BM25()
            # decoupe en token et indexe en meme temps
            index_bm25.index(bm25s.tokenize(texte_only))
            # sauvegarde les stats dans le chemin du dossier
            index_bm25.save(str(directory_bm25), corpus=texte_only)
        except (OSError, PermissionError, ValueError, TypeError) as e:
            print(f"Error: BM25 indexing failed ({e})")
            sys.exit(1)

        try:
            list_json: List[dict] = [chunk.model_dump() for chunk in all_chunk]
            json_file = directory_chunks / "chunks_data.json"

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(list_json, f, ensure_ascii=False, indent=4)
        except (PermissionError, OSError) as e:
            print(f"Error: while saving json: ({e})")
            sys.exit(1)
        print("Indexation over! Everything is saved!")
