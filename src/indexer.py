
from typing import List
from src.models import MinimalSource, ChunkStorage
import sys
from pathlib import Path
import bm25s
import json


class Create_chunk():

    def _chunk_data(self, full_data: str) -> List[str]:
        chunked_data: List[str] = []
        max_size = 2000
        start = 0
        while start < len(full_data):
            end = start + max_size
            chunk = full_data[start:end]
            chunked_data.append(chunk)
            start = end
        return chunked_data

    def read_then_chunk(self, path: str) -> List[ChunkStorage]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                full_data = f.read()
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error with the data file: {e}, End of programm")
            sys.exit(1)

        max_size: int = 2000
        chunked_data = self._chunk_data(full_data)
        files_sources = []
        for i, chunk in enumerate(chunked_data):
            start = i * max_size
            end = start + len(chunk)
            source = MinimalSource(
                file_path=path,
                first_character_index=start,
                last_character_index=end
            )
            full_info = ChunkStorage(
                source=source,
                text_content=chunk
            )
            files_sources.append(full_info)
        print(f"Chunk {i} succesfully created!")
        return files_sources

    def path_to_directory(self, path_dir: str) -> List[ChunkStorage]:

        main_directory = Path(path_dir)
        all_sources = []

        for file_path in main_directory.rglob("*"):

            if file_path.is_file() and file_path.suffix in [".py", ".md"]:
                print(f"File found : {file_path}")
                source_files = self.read_then_chunk(file_path)
                all_sources.extend(source_files)

        print(f"End of indexations. Chunk created: {len(all_sources)}")
        return all_sources

    def save_index(self, all_chunk: List[ChunkStorage],
                   destination_path: str = "data/processed"):
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
            # initialisation de la classe
            index_bm25 = bm25s.BM25()
            # decoupe en token et indexe en meme temps
            index_bm25.index(bm25s.tokenize(texte_only))
            # sauvegarde les stats dans le chemin du dossier
            index_bm25.save(str(directory_bm25), corpus=texte_only)
        except (OSError, PermissionError, ValueError, TypeError) as e:
            print("Error: a problem occured while attemping to index"
                  f"the file with bm25 ({e})")
            sys.exit(1)
        # on save dans un json
        try:
            list_json: List[dict] = [chunk.model_dump() for chunk in all_chunk]
            json_file = directory_chunks / "chunks_data.json"

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(list_json, f, ensure_ascii=False, indent=4)
        except (PermissionError, OSError) as e:
            print(f"Error: while saving json: ({e})")
            sys.exit(1)
        print("Indexation over ! Everything is saved!")
