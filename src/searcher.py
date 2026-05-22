import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import bm25s
from pydantic import BaseModel, Field
from src.models import MinimalSource
import re


def custom_tokenizer(text: str) -> list[str]:
    text = text.lower()
    # Cette regex capture les mots complets contenant des lettres
    return re.findall(r'[a-z0-9_]+', text)


class SearchSystem(BaseModel):
    """System to handle BM25 index loading and metadata sequence retrieval.

    This class complies with Pydantic validation rules requested
    by the project guidelines.
    """
    # le lambda evite d'avoir des datarace sur les chemin
    # il attend l'instanciation de la classe
    # et cree deux objet ath different
    storage_dir: Path = Field(
        default_factory=lambda: Path("data/processed")
    )
    bm25_dir: Path = Field(
        default_factory=lambda: Path("data/processed/bm25_index")
    )
    chunks_file: Path = Field(
        default_factory=lambda: Path("data/processed/chunks/chunks_data.json")
    )
    index_bm25: Any = None
    all_chunks_raw: List[Dict[str, Any]] = []

    class Config:
        """Pydantic configuration to allow arbitrary object types."""

        arbitrary_types_allowed = True

    def load_index_files(self, index_type: str = "docs") -> None:
        """Loads BM25 statistics and raw chunk metadata from the disk."""
        try:
            target_dir = self.storage_dir / f"bm25_index_{index_type}"
            
            self.index_bm25 = bm25s.BM25().load(
                str(target_dir), load_corpus=False
            )
            
            # 1. Charger tous les chunks bruts sauvés par l'indexeur
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
                
            # 2. ALIGNEMENT CRUCIAL : On ne garde que les chunks du même type !
            if index_type == "code":
                self.all_chunks_raw = [c for c in all_chunks if c["source"]["file_path"].endswith(".py")]
            else:
                self.all_chunks_raw = [c for c in all_chunks if not c["source"]["file_path"].endswith(".py")]

        except (ValueError, TypeError) as e:
            print(f"Error: Cannot load index file : {e}")
            sys.exit(1)
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error: cannot read index files ({e})")
            sys.exit(1)

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        """Queries the indexed corpus to extract the top-k relevant locations.

        Args:
            query: The user's query string in natural language.
            k: The maximum number of relevant documents to return.

        Returns:
            A list of type-validated MinimalSource Pydantic objects.
        """
        if self.index_bm25 is None or not self.all_chunks_raw:
            print(
                "Calling the loading function..."
            )
            self.load_index_files()

        try:
            # on transforme en token la question
            # nettoie avec regex
            tokens = custom_tokenizer(query)
            batch_tokens = [tokens]
            # calcul et retour sous forme de index et proba scores
            index, scores = self.index_bm25.retrieve(batch_tokens, k=k)
        except (ValueError, TypeError) as e:
            print(
                f"Error: problem while looking for the query in data: {e}"
            )
            return []

        retrieved_sources = []
        indices = (
                   index[0] if hasattr(index, "ndim") 
                   and index.ndim > 1 else index
        )
        for chunk_index in indices:
            full_chunk = self.all_chunks_raw[int(chunk_index)]
            # On reconstruit l'objet Pydantic
            final_src = MinimalSource(
                file_path=full_chunk["source"]["file_path"],
                first_character_index=(
                    full_chunk["source"]["first_character_index"]
                ),
                last_character_index=(
                    full_chunk["source"]["last_character_index"]
                ),
            )
            retrieved_sources.append(final_src)

        return retrieved_sources
