from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM

# create a custom DeepEval model by inheriting DeepEvalBaseLLM class
class HuggingFaceModel(DeepEvalBaseLLM):
    def __init__(
            self,
            model,
            tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        # the device to load the model onto
        # device = "cuda"
        device = "cpu"
        # device = "mps"

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        model = self.load_model()

        # the device to load the model onto
        # device = "cuda"
        device = "cpu"
        # device = "mps"

        model_inputs = self.tokenizer(prompts, return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)

    def get_model_name(self):
        return "Custom DeepEval Model"


def new_model(model_name):
    token = 'YOUR_HUGGING_FACE_ACCESS_TOKEN'

    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    return HuggingFaceModel(model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    model = new_model("distilbert/distilgpt2")
    print(model.generate("Tell me a joke"))
