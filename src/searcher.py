import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List
import bm25s
from pydantic import BaseModel, Field
from src.models import MinimalSource

# Liste des mots vides anglais les plus fréquents qui parasitent le BM25
STOPWORDS = {
    # Pronoms et déterminants
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "my",
    "your",
    "his",
    "her",
    "their",
    "our",
    "you",
    "i",
    "he",
    "she",
    "we",
    "they",
    "me",
    "him",
    "them",
    # Prépositions et connecteurs logiques
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "for",
    "with",
    "on",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "after",
    "before",
    "over",
    "under",
    "about",
    # Auxiliaires et verbes d'état fréquents
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "may",
    "might",
    "must",
    # Adverbes et particules génériques
    "not",
    "no",
    "yes",
    "very",
    "too",
    "so",
    "also",
    "just",
    "then",
    "there",
    "here",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "some",
    "such",
    "only",
}


def custom_tokenizer(text: str, is_code: bool = False) -> List[str]:
    """Sépare les mots de manière générique pour l'indexation."""
    sub_tokens = []
    if is_code:
        camel_tokens = re.findall(r"[A-Z][a-z0-9]+", text)
        sub_tokens.extend([c.lower() for c in camel_tokens if len(c) > 2])

    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", text)

    if is_code:
        for token in tokens:
            if "_" in token:
                sub_tokens.extend([t for t in token.split("_") if len(t) > 2])
        return tokens + sub_tokens

    stop_total = set(STOPWORDS)
    return [t for t in tokens if t not in stop_total]


class SearchSystem(BaseModel):
    """System to handle BM25 index loading and metadata sequence retrieval.

    This class complies with Pydantic validation rules requested
    by the project guidelines.
    """

    storage_dir: Path = Field(default_factory=lambda: Path("data/processed"))
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
                self.all_chunks_raw = [
                    c
                    for c in all_chunks
                    if c["source"]["file_path"].endswith(".py")
                ]
            else:
                self.all_chunks_raw = [
                    c
                    for c in all_chunks
                    if not c["source"]["file_path"].endswith(".py")
                ]

        except (ValueError, TypeError) as e:
            print(f"Error: Cannot load index file : {e}")
            sys.exit(1)
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error: cannot read index files ({e})")
            sys.exit(1)

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        """Performs a query search on the specified loaded BM25 index."""
        if self.index_bm25 is None or not self.all_chunks_raw:
            self.load_index_files()

        try:
            is_code_mode = self.index_type_meta == "code"

            if not is_code_mode:
                question_words = {"using", "command"}

                raw_words = query.lower().split()
                cleaned_words = [
                    w
                    for w in raw_words
                    if w.strip("?,.:;!") not in question_words
                ]

                if cleaned_words:
                    query = " ".join(cleaned_words)

            tokens = custom_tokenizer(query, is_code=is_code_mode)
            batch_tokens = [tokens]
            index, scores = self.index_bm25.retrieve(batch_tokens, k=k)
        except (ValueError, TypeError) as e:
            print(f"Error: problem while looking for the query in data: {e}")
            return []

        retrieved_sources = []
        indices = (
            index[0]
            if hasattr(index, "ndim") and index.ndim > 1
            else index
        )
        for chunk_index in indices:
            full_chunk = self.all_chunks_raw[int(chunk_index)]
            final_src = MinimalSource(
                file_path=full_chunk["source"]["file_path"],
                first_character_index=full_chunk["source"][
                    "first_character_index"
                ],
                last_character_index=full_chunk["source"][
                    "last_character_index"
                ],
            )
            retrieved_sources.append(final_src)

        return retrieved_sources
