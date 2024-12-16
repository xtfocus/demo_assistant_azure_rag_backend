import contextlib
import os

import fastapi
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from environs import Env
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncAzureOpenAI

from src.azure_container_client import AzureContainerClient
from src.get_pipeline import get_pipeline

from .globals import clients, configs, objects


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    from .config import ModelConfig

    configs["app_config"] = ModelConfig()

    config = configs["app_config"]

    credential = AzureKeyCredential(config.AZURE_SEARCH_ADMIN_KEY)

    clients["chat-completion-model"] = AsyncAzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        timeout=config.timeout,
        max_retries=config.retry_attempts,
    )

    clients["blob_service_client"] = BlobServiceClient.from_connection_string(
        config.AZURE_STORAGE_CONNECTION_STRING
    )

    # SEARCH AND STORAGE RESOURCE NEEDED TO FIND AND DELETE OLD RECORDS
    clients["image_container_client"] = AzureContainerClient(
        client=clients["blob_service_client"],
        container_name=config.IMAGE_CONTAINER_NAME,
    )
    # azure ai search clients
    clients["text-azure-ai-search"] = SearchClient(
        config.AZURE_SEARCH_SERVICE_ENDPOINT,
        config.TEXT_INDEX_NAME,
        credential=credential,
    )
    clients["image-azure-ai-search"] = SearchClient(
        config.AZURE_SEARCH_SERVICE_ENDPOINT,
        config.IMAGE_INDEX_NAME,
        credential=credential,
    )
    clients["summary-azure-ai-search"] = SearchClient(
        config.AZURE_SEARCH_SERVICE_ENDPOINT,
        config.SUMMARY_INDEX_NAME,
        credential=credential,
    )

    # PIPELINE object
    objects["pipeline"] = get_pipeline(
        configs["app_config"],
        clients["chat-completion-model"],
        clients["image_container_client"],
    )

    yield

    clients["blob_service_client"].close()
    await clients["chat-completion-model"].close()
    clients["text-azure-ai-search"].close()
    clients["image-azure-ai-search"].close()


def create_app():
    env = Env()

    if not os.getenv("RUNNING_IN_PRODUCTION"):
        env.read_env(".env.hub")

    app = fastapi.FastAPI(docs_url="/", lifespan=lifespan)

    origins = env.list("ALLOWED_ORIGINS", ["http://localhost", "http://localhost:8080"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from . import main

    app.include_router(main.router)

    return app
