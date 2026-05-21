import bm25s
import sys
import json
from pathlib import Path
from typing import List
from src.models import MinimalSource


class SearchSystem():
    def __init__(self, storage_dir: str = "data/processed"):
        self.storage_dir = Path(storage_dir)
        self.bm25_dir = self.storage_dir / "bm25_index"
        self.chunks_file = self.storage_dir / "chunks" / "chunks_data.json"

        self.index_bm25 = None
        self.all_chunks_raw = []

    def load_index_files(self):
        try:
            # instanciation 
            # il recharge les fichier de stats
            # load_corpus = charger le texte associee pas seulement les stats
            # il recupere les truc quon a save plus tot avec save index
            self.index_bm25 = bm25s.BM25.load(str(self.bm25_dir),
                                              load_corpus=False)
            # ouvrir le dico des index 
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                self.all_chunks_raw = json.load(f)

        except (ValueError, TypeError) as e:
            print(f"Error: Cannot load index file : {e}")
            sys.exit(1)
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error: cannot read index files ({e})")
            sys.exit(1)

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        if self.index_bm25 is None or not self.all_chunks_raw:
            print("Error: Empty index file or file not loaded,"
                  "trying to call the loading function...")
            self.load_index_files()

        try:
            # on transforme en token la question
            query_token = bm25s.tokenize([query])
            # calcul et retour sous forme de index et proba scores
            index, scores = self.index_bm25.retrieve(query_token, k=k)
        except (ValueError, TypeError) as e:
            print(f"Error: problem while looking for the query in data: {e}")
            return []

        retrieved_sources = []
        for chunk_index in index[0]:
            full_chunk = self.all_chunks_raw[chunk_index]
        # On reconstruit l'objet Pydantic
            final_src = MinimalSource(
                file_path=full_chunk["source"]["file_path"],
                first_character_index=(full_chunk["source"]
                                                 ["first_character_index"]),
                last_character_index=(full_chunk["source"]
                                                ["last_character_index"])
            )
            retrieved_sources.append(final_src)

        return retrieved_sources
