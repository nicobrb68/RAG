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

    def load_index_files(self) -> None:
        """Loads BM25 statistics and raw chunk metadata from the disk."""
        try:
            # instanciation
            # il recharge les fichier de stats
            # load_corpus = charger le texte associee pas seulement les stats
            # il recupere les truc quon a save plus tot avec save index
            self.index_bm25 = bm25s.BM25(k1=1.2, b=0.8).load(
                str(self.bm25_dir), load_corpus=False
            )
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
