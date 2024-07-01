import os
from dotenv import load_dotenv
from langchain_google_vertexai import (
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory
)
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.messages.human import HumanMessage

class GeminiModel(DeepEvalBaseLLM):
    """Class to implement Vertex AI Gemini model for DeepEval"""
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Vertex AI Model"

    def generate_samples(self, prompt: str, n: int, temperature: float):
        chat_model = self.load_model()

        print("temp: ", temperature)
        chat_model.temperature = temperature

        print("n :", n)
        chat_model.n = 1  # gemini doesn't support generation of multiple samples per prompt, so we will iterate manually

        all_completions = []

        for _ in range(10):
            generations = chat_model._generate([HumanMessage(prompt)]).generations
            completions = [r.text for r in generations]
            all_completions.extend(completions)

        print("completions: ", all_completions)
        return all_completions


def new_model(model_name):
    # Initialize safety filters for vertex model - to ensure no evaluation responses are blocked
    safety_settings = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Load the environment variables from .env file
    load_dotenv()
    GCP_PROJECT = os.getenv("GCP_PROJECT")
    GCP_LOCATION = os.getenv("GCP_LOCATION")

    custom_model_gemini = ChatVertexAI(
        model_name=model_name,
        safety_settings=safety_settings,
        project=GCP_PROJECT,
        location=GCP_LOCATION
    )

    # Initialize the wrapper class
    vertexai_gemini = GeminiModel(model=custom_model_gemini)

    return vertexai_gemini


if __name__ == '__main__':
    model = new_model("gemini-1.5-pro-001")
    print(model.generate("Tell me a joke"))
