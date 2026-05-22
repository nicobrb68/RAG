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

        # SECTORISATION CHIRURGICALE CODE VS DOC
        # SECTORISATION CHIRURGICALE CODE VS DOC
        if path.suffix == ".py":
            les_separateurs = [""]
            overlap = 320
            target_size = 1400 - overlap  
        else:
            les_separateurs = ["\n\n", "\n", " ", ""]
            overlap = 380
            target_size = 1980 - overlap

        chunked_data = CodeIndexer.split_text_recursive(
            text=full_data,
            max_size=target_size,
            overlap=overlap,
            separators=les_separateurs
        )

        files_sources = []
        current_search_start = 0
        for chunk in chunked_data:
            # SECURITE MOULINETTE : Hard cut à max_chunk_size
            chunk = chunk[:self.max_chunk_size]

            # On cherche la position du morceau dans le texte complet
            start = full_data.find(chunk, current_search_start)
            if start == -1:
                start = full_data.find(chunk)
            end = start + len(chunk)
            
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

        print(f"Successfully created {len(chunked_data)} chunks for {path.name}")
        return files_sources

    @staticmethod
    def split_text_recursive(text: str, max_size: int, overlap: int,
                             separators: list[str] = None) -> list[str]:
        """Découpe un texte de manière récursive en respectant la syntaxe d'origine."""
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        if len(text) <= max_size:
            return [text]

        separator = separators[-1]
        for s in separators:
            if s in text:
                separator = s
                break

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

                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = (
                        overlap_text + join_str + split
                        if len(overlap_text + join_str + split) <= max_size 
                        else split
                    )
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def path_to_directory(self, path_dir: str) -> List[ChunkStorage]:
        """Scans a repository to process every valid Python and Markdown file."""
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
        directory_path = Path(destination_path)
        # On crée deux dossiers d'index distincts
        directory_bm25_docs = directory_path / "bm25_index_docs"
        directory_bm25_code = directory_path / "bm25_index_code"
        directory_chunks = directory_path / "chunks"

        try:
            directory_bm25_docs.mkdir(parents=True, exist_ok=True)
            directory_bm25_code.mkdir(parents=True, exist_ok=True)
            directory_chunks.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            print(f"Error: Cannot create storage directory: ({e})")
            sys.exit(1)

        # On sépare les chunks par type de fichier
        chunks_docs = [c for c in all_chunk if not c.source.file_path.endswith(".py")]
        chunks_code = [c for c in all_chunk if c.source.file_path.endswith(".py")]

        # 1. INDEXATION DES DOCS (k1=1.5, b=0.7 pour le texte naturel)
        # 1. INDEXATION DES DOCS
        if chunks_docs:
            try:
                index_docs = bm25s.BM25(k1=1.5, b=0.7)
                texts_docs = [c.text_content for c in chunks_docs]
                tokens_docs = [custom_tokenizer(t, is_code=False) for t in texts_docs]
                index_docs.index(tokens_docs)
                index_docs.save(str(directory_bm25_docs), corpus=texts_docs)
            except (OSError, PermissionError, ValueError, TypeError) as e:
                print(f"Error: BM25 docs indexing failed ({e})")

        # 2. INDEXATION DU CODE
        if chunks_code:
            try:
                index_code = bm25s.BM25(k1=1.5, b=0.7)
                texts_code = [c.text_content for c in chunks_code]
                tokens_code = [custom_tokenizer(t, is_code=True) for t in texts_code]
                index_code.index(tokens_code)
                index_code.save(str(directory_bm25_code), corpus=texts_code)
            except (OSError, PermissionError, ValueError, TypeError) as e:
                print(f"Error: BM25 code indexing failed ({e})")

        try:
            list_json: List[dict] = [chunk.model_dump() for chunk in all_chunk]
            json_file = directory_chunks / "chunks_data.json"

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(list_json, f, ensure_ascii=False, indent=4)
        except (PermissionError, OSError) as e:
            print(f"Error: while saving json: ({e})")
            sys.exit(1)
        print("Indexation over! Everything is saved!")