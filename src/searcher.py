import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import bm25s
from pydantic import BaseModel, Field
from src.models import MinimalSource
import re

def custom_tokenizer(text: str, is_code: bool = False) -> list[str]:
    # 1. Si c'est du code, on extrait d'abord les sous-mots CamelCase AVANT le lower()
    sub_tokens = []
    if is_code:
        # Trouve les transitions de majuscules (ex: FlashAttention -> Flash, Attention)
        camel_tokens = re.findall(r'[A-Z][a-z0-9]+', text)
        sub_tokens.extend([c.lower() for c in camel_tokens if len(c) > 2])

    text = text.lower()
    tokens = re.findall(r'[a-z0-9_]+', text)
    
    # 2. On éclate le Snake Case comme tout à l'heure
    if is_code:
        for token in tokens:
            if "_" in token:
                sub_tokens.extend([t for t in token.split("_") if len(t) > 2])
        return tokens + sub_tokens
        
    return tokens


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
    index_type_meta: str = "docs"

    class Config:
        """Pydantic configuration to allow arbitrary object types."""

        arbitrary_types_allowed = True

    def load_index_files(self, index_type: str = "docs") -> None:
        """Loads BM25 statistics and raw chunk metadata from the disk."""
        try:
            self.index_type_meta = index_type
            target_dir = self.storage_dir / f"bm25_index_{index_type}"
            
            self.index_bm25 = bm25s.BM25().load(
                str(target_dir), load_corpus=False
            )

            with open(self.chunks_file, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)

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
            # On passe l'information du type d'index au tokenizer
            is_code_mode = (self.index_type_meta == "code")
            tokens = custom_tokenizer(query, is_code=is_code_mode)
            
            batch_tokens = [tokens] 
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
