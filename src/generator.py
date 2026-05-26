import sys
from typing import Any, List
import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class AnswerGenerator(BaseModel):
    """LLM-based generator for context-aware answers optimized for CPU."""

    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "cpu"
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)

    class Config:
        """Pydantic configuration to allow arbitrary object types."""

        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize the model and tokenizer after Pydantic validation."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        except OSError as e:
            print(
                f"RuntimeError: Failed to load model '{self.model_name}'. "
                f"Details: {e}",
                file=sys.stderr,
            )

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """Generate an answer for a single query by leveraging batching."""
        answers = self.generate_answers_batch([query], [contexts])
        return answers[0] if answers else "Information not found"

    def generate_answers_batch(
        self, queries: List[str], contexts_list: List[List[str]]
    ) -> List[str]:
        """Generate answers for a batch of queries simultaneously for CPU."""
        if not queries or self.model is None or self.tokenizer is None:
            return []

        formatted_prompts = []
        for query, context_chunks in zip(queries, contexts_list):
            cleaned_chunks = [c.strip() for c in context_chunks if c.strip()]
            full_context = "\n---\n".join(cleaned_chunks)

            # Raccourci drastique du contexte pour éviter de noyer Qwen3-0.6B
            context_str = full_context[:1200]
            
            # Prompt direct sans Chat Template (les petits modèles se perdent dedans)
            text = (
                f"Task: Extract the exact technical terms from the context "
                f"to answer the question.\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n"
                f"Direct Technical Answer:"
            )
            formatted_prompts.append(text)

        model_inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                max_new_tokens=60,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids
            )
        ]

        decoded_outputs = self.tokenizer.batch_decode(
            new_generated_ids, skip_special_tokens=True
        )

        processed_answers = []
        for output in decoded_outputs:
            answer_text = output.strip()
            
            # Nettoyage des retours à la ligne pour le format de la moulinette
            answer_text = answer_text.split("\n")[0]
            answer_text = answer_text.replace("\n", " ").strip()
            answer_text = answer_text.strip('"').strip("'").strip()

            # Extraction si le modèle répète le tag de départ
            if "technical answer:" in answer_text.lower():
                answer_text = (
                    answer_text.lower()
                    .split("technical answer:")[-1]
                    .strip()
                )

            processed_answers.append(
                answer_text if answer_text else "Information not found"
            )

        return processed_answers