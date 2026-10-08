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
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_fallbacks: str = ("google/gemma-4-26b-a4b-it:free,"
                                 "nvidia/nemotron-3-ultra-550b-a55b:free")

    # Retrieval
    retriever_k: int = 5

        # Build device (indexing side only; serving always CPU)
    device: str = "cpu"  # flip to "cuda" for GPU index builds
    use_fp16: bool = False  # True with cuda on 4GB cards (halves VRAM)
    embed_batch_size: int = 1000  # progress print per batch

    # Versioned index + hybrid retrieval
    index_dir: str = "data/processed/faiss_index_v2"
    use_bm25: bool = False
    rrf_k: int = 60  # reciprocal-rank-fusion constant

    processed_dir: str = "data/processed"  # chunks source (index_dir holds built artifacts)


settings = Settings()