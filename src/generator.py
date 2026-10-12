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
    def __init__(self) -> None:
        self.model_name = "Qwen/Qwen3-0.6B"

        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float32
        )
        self.model.eval()

    @staticmethod
    def bulid_context(chunks: list[FullSource] | list[MinimalSource]) -> str:
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

        with open(student_search_results_path) as f:
            dataset = StudentSearchResults.model_validate_json(f.read())

        answers = []

        for s in tqdm(dataset.search_results, desc="Answering"):
            context = self.bulid_context(s.retrieved_sources[:3])
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
        # prepare the model input
        role = (
            "Answer the question directly using ONLY the provided context.\n"
            "Rules:\n"
            "- If the answer cannot be found in the context, respond EXACTLY "
            "with: 'Information not found in the context.'\n"
            "- Do not use outside knowledge or make assumptions.\n"
            "- Be concise, factual, and to the point (1 to 2 sentences).\n"
            "- Extract exact values, flags, class names, or HTTP endpoints "
            "directly from the text."
        )
        user_ = f"Context:\n{context}\nQuestion:\n{prompt}\n Direct Answer:"

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

        # conduct text completion
        with torch.inference_mode():
            generated_ids = self.model.generate(  # type: ignore[misc]
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.1,
            )
        output_ids = generated_ids[0][
            len(model_inputs.input_ids[0]):
        ].tolist()

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if isinstance(content, list):
            content = " ".join(content)
        return str(content)
