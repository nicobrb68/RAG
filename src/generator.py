import re
import sys
from typing import Any, List
from llama_cpp import Llama
from pydantic import BaseModel, Field


class AnswerGenerator(BaseModel):
    """Loads Qwen3-0.6B via llama.cpp for ultra-fast CPU inference."""

    model_path: str = "data/models/qwen3-0.6b.gguf"
    max_new_tokens: int = 512  # La limite haute de ton pote
    max_context_chars: int = 6000
    llm: Any = Field(default=None, exclude=True)

    class Config:
        """Pydantic configuration to allow arbitrary object types."""

        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize the native C++ Llama engine after validation."""
        try:
            # Charge le modèle GGUF instantanément en mémoire CPU
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,     # Assez grand pour encaisser la doc
                n_threads=4,    # Monte à 6 ou 8 si tu as un gros processeur
                verbose=False,
            )
        except Exception as e:
            print(
                f"RuntimeError: Failed to load Llama-CPP model. "
                f"Details: {e}",
                file=sys.stderr,
            )

    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Let the AI reason and extract the technical answer using C++."""
        if not self.llm or not contexts:
            return "Information not found."

        # Reconstruction du contexte à la lettre comme ton pote
        context_parts = []
        total_chars = 0
        for chunk_text in contexts:
            if not chunk_text:
                continue
            if total_chars + len(chunk_text) > self.max_context_chars:
                break
            context_parts.append(f"Source: chunk\n{chunk_text}")
            total_chars += len(chunk_text)

        context = "\n\n---\n\n".join(context_parts)

        # Structure de messages de ton pote
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions about "
                    "the vLLM codebase. Answer based ONLY on the provided "
                    "sources. Be concise and self-contained. Mention the "
                    "source file(s) you draw from."
                ),
            },
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {question}",
            },
        ]

        try:
            # Inférence native en C++ (vitesse maximale sur CPU)
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                repeat_penalty=1.3,  # La pénalité anti-boucle de ton pote
            )
            
            answer = response["choices"][0]["message"]["content"].strip()
            
            # Nettoyage regex strict des balises de réflexion de ton pote
            answer = re.sub(
                r"<think>.*?</think>", "", answer, flags=re.DOTALL
            ).strip()
            
            return answer if answer else "Information not found."
            
        except Exception:
            return "Information not found."

    def generate_answers_batch(
        self, queries: List[str], contexts_list: List[List[str]]
    ) -> List[str]:
        """Process queries sequentially using the fast native C++ engine."""
        return [
            self.generate_answer(q, c) for q, c in zip(queries, contexts_list)
        ]