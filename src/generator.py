import os
import re
import sys
from typing import Any, Dict, List, Optional
from llama_cpp import Llama
from pydantic import BaseModel, Field


class AnswerGenerator(BaseModel):
    """Loads Qwen3-0.6B via llama.cpp with automatic download fallback."""

    model_dir: str = "data/models"
    model_name: str = "qwen3-0.6b.gguf"
    max_new_tokens: int = 380
    max_context_chars: int = 4500
    llm: Optional[Llama] = Field(default=None, exclude=True)

    class Config:
        """Pydantic configuration to allow arbitrary object types."""

        arbitrary_types_allowed = True

    @property
    def model_path(self) -> str:
        """Get the full path to the model file."""
        return os.path.join(self.model_dir, self.model_name)

    def model_post_init(self, __context: Any) -> None:
        """Ensure the model exists (or download it) and initialize Llama."""
        os.makedirs(self.model_dir, exist_ok=True)

        if not os.path.exists(self.model_path):
            print(
                f"Model not found at {self.model_path}.\n"
                f"Downloading Qwen GGUF from Hugging Face... Please wait...",
                file=sys.stderr,
            )
            try:
                import urllib.request

                url = (
                    "https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/"
                    "resolve/main/qwen1_5-0_5b-chat-q4_k_m.gguf"
                )
                urllib.request.urlretrieve(url, self.model_path)
                print("Download complete success!", file=sys.stderr)
            except Exception as e:
                print(
                    f"RuntimeError: Automated download failed: {e}",
                    file=sys.stderr,
                )
                return

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=4,
                flash_attn=True,
                verbose=False,
            )
        except Exception as e:
            print(
                f"RuntimeError: Failed to load Llama-CPP model: {e}",
                file=sys.stderr,
            )

    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate an answer letting the AI fully express itself."""
        if not self.llm or not contexts:
            return "Information not found."

        context_parts: List[str] = [
            f"Source: chunk\n{c.strip()}" for c in contexts if c.strip()
        ]
        context: str = "\n\n---\n\n".join(context_parts)[
            : self.max_context_chars
        ]

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions about "
                    "the vLLM codebase. Answer based ONLY on the provided "
                    "sources. Provide a complete, concise, and well-written "
                    "answer in plain English using full sentences."
                ),
            },
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {question}",
            },
        ]

        try:
            # Typage explicite du dictionnaire de retour de llama_cpp
            response: Dict[str, Any] = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                repeat_penalty=1.3,
            )

            answer: str = response["choices"][0]["message"]["content"].strip()

            # On enlève juste la chaîne de pensée interne
            answer = re.sub(
                r"<think>.*?</think>", "", answer, flags=re.DOTALL
            ).strip()

            # Nettoyage minimal des caractères bizarres et espaces
            answer = re.sub(r"[^\x00-\x7F]+", "", answer).strip()
            answer = answer.strip(':,.-"\' `')

            if answer:
                answer = answer[:1].upper() + answer[1:]
                if not answer.endswith("."):
                    ans_lower: str = answer.lower()
                    if not (
                        ans_lower.endswith(("=", "-", "_"))
                        or ans_lower.split()[-1].startswith("-")
                    ):
                        answer += "."
            else:
                answer = "Information not found."

            return answer

        except Exception:
            return "Information not found."

    def generate_answers_batch(
        self, queries: List[str], contexts_list: List[List[str]]
    ) -> List[str]:
        """Synchronous evaluation loop for reliability."""
        return [
            self.generate_answer(q, c) for q, c in zip(queries, contexts_list)
        ]
