from tqdm_test import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .models import MinimalAnswer, StudentSearchResults, StudentSearchResultsAndAnswer


class Generator:
    def __init__(self):
        self.model_name = "Qwen/Qwen3-0.6B"

        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )

    @staticmethod
    def bulid_context(chunks):
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
            context = self.bulid_context(s.retrieved_sources[:5])
            ans = self.generate_answer(context, s.question)

            answers.append(
                MinimalAnswer(
                    question_id=s.question_id,
                    question=s.question,
                    retrieved_sources=s.retrieved_sources,
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

    def generate_answer(self, context, prompt):

        # prepare the model input
        role = f"""You are a helpful assistant for vLLM.
Extract the concise answer to the user question using ONLY the provided documentation snippets.
Only if the snippets contain zero relevant information, reply with: I don't know.
Documentation:
{context}
    """

        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # conduct text completion
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=128,
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        # print("thinking content:", thinking_content)
        # print("content:", content)
        return content
