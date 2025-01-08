from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from google.generativeai import GeminiClient
# Import Gemini SDK

class GeminiLLM(BaseLLM):
    """Custom LLM wrapper for Gemini."""

    def __init__(self, model_name: str, temperature: float, api_key: str):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

    def _call(self, prompt: str, stop: list = None) -> str:
        """Call the Gemini API to generate text."""
        client = GeminiClient(api_key=self.api_key)
        try:
            response = client.generate_text(prompt, model=self.model_name, temperature=self.temperature)
            return response.get("text", "")
        except Exception as e:
            raise RuntimeError(f"Failed to generate text using Gemini: {e}")

    @property
    def _llm_type(self) -> str:
        return "gemini"


class GeminiEmbeddings(Embeddings):
    """Custom embeddings wrapper for Gemini."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def embed_documents(self, documents: list) -> list:
        """Generate embeddings for a list of documents."""
        client = GeminiClient(api_key=self.api_key)
        try:
            embeddings = [client.embed(doc).get("embedding", []) for doc in documents]
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to embed documents using Gemini: {e}")

    def embed_query(self, query: str) -> list:
        """Generate an embedding for a single query."""
        client = GeminiClient(api_key=self.api_key)
        try:
            return client.embed(query).get("embedding", [])
        except Exception as e:
            raise RuntimeError(f"Failed to embed query using Gemini: {e}")
