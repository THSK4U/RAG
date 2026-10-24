import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import (
    FullSource,
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class Generator:
    """Answer generator using a language model over retrieved context."""

    def __init__(self) -> None:
        """Initialize tokenizer and model for answer generation."""
        self.model_name = "Qwen/Qwen3-0.6B"

        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float32
        )
        self.model.eval()

    @staticmethod
    def bulid_context(chunks: list[FullSource] | list[MinimalSource]) -> str:
        """Build concatenated context text from source chunks.

        Args:
            chunks: List of retrieved source chunks.

        Returns:
            str: Combined context text with file paths.
        """
        parts = []
        for chunk in chunks:
            first_char = chunk.first_character_index
            last_char = chunk.last_character_index
            with open(chunk.file_path) as f:
                text = f.read()[first_char:last_char]
            parts.append(f"[{chunk.file_path}]\n{text}")
        context = "\n\n".join(parts)
        return context

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for a dataset of search results.

        Args:
            student_search_results_path: Path to input search results JSON.
            save_directory: Destination directory to save answer results JSON.
        """
        with open(student_search_results_path) as f:
            dataset = StudentSearchResults.model_validate_json(f.read())

        answers = []

        for s in tqdm(dataset.search_results, desc="Answering"):
            context = self.bulid_context(s.retrieved_sources)
            ans = self.generate_answer(context, s.question)
            retriver_convert: list[FullSource | MinimalSource] = [
                MinimalSource.model_validate(src, from_attributes=True)
                for src in s.retrieved_sources
            ]
            answers.append(
                MinimalAnswer(
                    question_id=s.question_id,
                    question=s.question,
                    retrieved_sources=retriver_convert,
                    answer=ans,
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=dataset.k,
        )

        import os

        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(student_search_results_path)
        output_path = os.path.join(save_directory, filename)
        with open(output_path, "w") as f:
            f.write(output.model_dump_json(indent=2))

        print("DONE!")

    def generate_answer(self, context: str, prompt: str) -> str:
        """Generate an answer using context and query.

        Args:
            context: Retrieved context text.
            prompt: User question or query.

        Returns:
            str: Generated concise answer.
        """

        role = (
            "Answer using ONLY the provided context.\n"
            "Rules:\n"
            "- If the answer cannot be found in the context, respond EXACTLY "
            "with: 'Information not found in the context.'\n"
            "- Do not use outside knowledge or make assumptions.\n"
            "- Be concise, factual, and to the point (1 to 2 sentences).\n"
            "- Extract exact values, flags, class names, or HTTP endpoints "
            "directly from the text."
        )
        user_ = (
            f"Question: {prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Based on the context above, answer the question: {prompt}\n"
            f"Answer:"
        )
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": user_},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(
            self.model.device
        )

        # Greedy deterministic decoding
        with torch.inference_mode():
            generated_ids = self.model.generate(  # type: ignore[misc]
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
            )
        output_ids = generated_ids[0][
            len(model_inputs.input_ids[0]):
        ].tolist()

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if isinstance(content, list):
            content = " ".join(content)
        return str(content)
