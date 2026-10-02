from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Turath AI"
    debug: bool = False

    # HuggingFace (embeddings now, LLM later)
    hf_token: str = ""
    embedding_model: str = "BAAI/bge-m3"

    # Paths
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"

    model_config = {"env_file": ".env", "extra": "ignore"}


    # OpenRouter LLM (generation)
    openrouter_api_key: str = ""
    openrouter_model: str = "qwen/qwen3.8-27b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_fallbacks: str = ("google/gemma-4-26b-a4b-it:free,"
                                 "nvidia/nemotron-3-ultra-550b-a55b:free")

    # Retrieval
    retriever_k: int = 5


settings = Settings()