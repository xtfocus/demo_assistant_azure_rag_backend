import os
from dataclasses import dataclass


@dataclass
class GlobalAppConfig:
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 4096
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 0.5

    ALGORITHM_CONFIGURATION_NAME = os.getenv("ALGORITHM_CONFIGURATION_NAME", "myHnsw")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-05-01-preview"
    )
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embedding-model"
    )
    AZURE_OPENAI_EMBEDDING_DIMENSIONS = int(
        os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 1536)
    )
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_MODEL_NAME = os.getenv(
        "AZURE_OPENAI_MODEL_NAME", "text-embedding-3-large"
    )
    AZURE_SEARCH_ADMIN_KEY: str = os.getenv("AZURE_SEARCH_ADMIN_KEY", "")
    AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", "")
    AZURE_STORAGE_CONNECTION_STRING: str = os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING", ""
    )
    MODEL_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    SEMANTIC_CONFIGURATION_NAME = os.getenv(
        "SEMANTIC_CONFIGURATION_NAME", "my-semantic-config"
    )
    VECTORIZER_NAME = os.getenv("VECTORIZER_NAME", "myHnswProfile")
    VECTOR_SEARCH_PROFILE_NAME = os.getenv(
        "VECTOR_SEARCH_PROFILE_NAME", "myHnswProfile"
    )

    TEXT_INDEX_NAME = os.getenv("TEXT_INDEX_NAME", "mc-text-index")
    IMAGE_INDEX_NAME = os.getenv("IMAGE_INDEX_NAME", "mc-image-index")
    SUMMARY_INDEX_NAME = os.getenv("SUMMARY_INDEX_NAME", "mc-summary-index")

    IMAGE_CONTAINER_NAME = os.getenv("IMAGE_CONTAINER_NAME", "my-image-container")
