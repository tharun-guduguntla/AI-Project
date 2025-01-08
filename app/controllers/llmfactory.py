import yaml
from app.integrations.gemini_integration import GeminiLLM, GeminiEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

class LLMFactory:
    """Factory for creating LLM and Embedding instances based on configuration."""

    def __init__(self, config, provider_name):
        self.config = config
        self.provider_name = provider_name

    def get_llm(self):
        """Returns the appropriate LLM instance based on the provider."""
        provider_config = self.config[f"{self.provider_name.lower()}llm_provider"]
        if provider_config["name"] == "OpenAI":
            return ChatOpenAI(
                model_name=provider_config["model_settings"]["model_name"],
                temperature=provider_config["model_settings"]["temperature"],
                api_key=provider_config["api_keys"],
            )
        elif provider_config["name"] == "gemini":
            return GeminiLLM(
                model_name=provider_config["model_settings"]["model_name"],
                temperature=provider_config["model_settings"]["temperature"],
                api_key=provider_config["api_keys"],
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_config['name']}")

    def get_embeddings(self):
        """Returns the appropriate embeddings instance based on the provider."""
        provider_config = self.config[f"{self.provider_name.lower()}llm_provider"]
        if provider_config["name"] == "OpenAI":
            return OpenAIEmbeddings(api_key=provider_config["api_keys"])
        elif provider_config["name"] == "gemini":
            return GeminiEmbeddings(api_key=provider_config["api_keys"])
        else:
            raise ValueError(f"Unsupported Embedding provider: {provider_config['name']}")