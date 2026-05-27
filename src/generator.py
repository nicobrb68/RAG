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

        if self.tokenizer:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate an answer for a single query by leveraging batching."""
        answers = self.generate_answers_batch([question], [contexts])
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
            context_str = " ".join(cleaned_chunks)[:700]

            # Application des consignes du pote au format "Modèle Base"
            text = (
                f"Instructions: Answer the question using only the facts "
                f"from the document. Be direct, faithful and relevant.\n\n"
                f"Document:\n{context_str}\n\n"
                f"Question: {query.strip(' ?')}\n"
                f"Answer: According to vLLM documentation, the exact "
                f"technical solution is"
            )
            formatted_prompts.append(text)

        model_inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1200,
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                max_new_tokens=35,
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
        for output, query, context_chunks in zip(
            decoded_outputs, queries, contexts_list
        ):
            ans = output.strip().split("\n")[0].strip()
            
            # Ne coupe pas les versions décimales comme 3.9
            if ". " in ans:
                ans = ans.split(". ")[0].strip()

            bad_patterns = [
                "[list", "___", "__", "{", "}", "put the answer", 
                "box", "href", "insert your answer", "[blank]", "above"
            ]

            # Fallback rigoureux par extraction si le modèle déraille
            if (
                not ans
                or len(ans) < 3
                or any(pat in ans.lower() for pat in bad_patterns)
            ):
                keywords = [
                    w.strip("?,.!") for w in query.lower().split() if len(w) > 4
                ]
                extracted = ""
                for sentence in " ".join(context_chunks).split(". "):
                    if len(sentence.strip()) > 25 and any(
                        k in sentence.lower() for k in keywords
                    ):
                        extracted = sentence.strip()
                        break
                ans = extracted if extracted else "Information not found"

            prefix = "According to vLLM documentation, the exact technical solution is "
            if not ans.lower().startswith("according"):
                ans = prefix + ans[:1].lower() + ans[1:]

            if ans and not ans.endswith("."):
                ans += "."

            processed_answers.append(" ".join(ans.split()))

        return processed_answers