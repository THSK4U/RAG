from transformers import AutoModelForCausalLM, AutoTokenizer


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

    def generate_answer(self, context, prompt):

        # prepare the model input
        role = f"""You are a technical assistant for the vLLM codebase.
    Answer using ONLY the sources below.
    If the answer is not there, say: I don't know.
    Sources:\n{context}
    """

        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=32768)
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
