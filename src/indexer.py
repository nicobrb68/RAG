import sys
import json
from pathlib import Path
from typing import List
import bm25s
from pydantic import BaseModel
from src.models import MinimalSource, ChunkStorage
from src.searcher import custom_tokenizer


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

    def _read_then_chunk(self, path: Path) -> List[ChunkStorage]:
        """Reads a specific file path and creates localized Pydantic chunks."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                full_data = f.read()
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error with the data file: {e}, End of program")
            sys.exit(1)

        # 1. CHOIX DES SÉPARATEURS SELON L'EXTENSION 
        if path.suffix == ".py":
            les_separateurs = [""]
        else:
            les_separateurs = ["\n\n", "\n", " ", ""]

        # Ajustement pour respecter la limite de 2000 caractères avec overlap
        overlap = 200
        target_size = self.max_chunk_size - overlap

        chunked_data = CodeIndexer.split_text_recursive(
            text=full_data,
            max_size=target_size,
            overlap=overlap,
            separators=les_separateurs
        )

        files_sources = []

        current_search_start = 0
        for chunk in chunked_data:
            # SECURITE: On force la coupure à max_chunk_size pour la moulinette
            chunk = chunk[:self.max_chunk_size]

            # On cherche la position du morceau dans le texte complet
            start = full_data.find(chunk, current_search_start)
            if start == -1:
                # Sécurité au cas où
                start = full_data.find(chunk)

            end = start + len(chunk)
            # on garde un overlap de 200
            current_search_start = start + max(1, len(chunk) - overlap)

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

        print(f"Successfully created {len(chunked_data)} chunks "
              f"for {path.name}")
        return files_sources

    @staticmethod
    def split_text_recursive(text: str, max_size: int, overlap: int,
                             separators: list[str] = None) -> list[str]:
        """Découpe un texte de manière récursive en respectant
        la syntaxe et les séparateurs.
        """
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        if len(text) <= max_size:
            return [text]

        separator = separators[-1]
        for s in separators:
            if s in text:
                separator = s
                break

        # Découper texte selon séparateur
        splits = text.split(separator) if separator != "" else list(text)

        chunks = []
        current_chunk = ""

        for split in splits:
            join_str = separator if current_chunk else ""
            potential_chunk = current_chunk + join_str + split

            if len(potential_chunk) <= max_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # on overlap les chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    # Sécurité pour éviter les boucles infinies
                    current_chunk = (
                                     overlap_text + join_str + split
                                     if len(overlap_text + join_str + split) 
                                     <= max_size else split
                    )
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def path_to_directory(self, path_dir: str) -> List[ChunkStorage]:
        """Scans a repository to process every valid Python and
        Markdown file."""
        main_directory = Path(path_dir)
        all_sources = []

        for file_path in main_directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".py", ".md"]:
                print(f"File found : {file_path}")
                source_files = self._read_then_chunk(file_path)
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
            # 1. Configuration des hyperparamètres BM25 optimisés pour les extra-credits
            index_bm25 = bm25s.BM25(k1=1.2, b=0.8)

            tokenized_corpus = [custom_tokenizer(text) for text in texte_only]

            # Indexation et sauvegarde
            index_bm25.index(tokenized_corpus)
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