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


settings = Settings()