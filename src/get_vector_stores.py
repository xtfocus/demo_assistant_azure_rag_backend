"""
Create (or get existing) text and image Azure search indexes
"""

from src.azure_service_integration.search_objects import (get_semantic_search,
                                                          get_vector_search)
from src.azure_service_integration.vector_stores import (
    MyAzureOpenAIEmbeddings, MyAzureSearch)
from src.fields import get_fields


def get_vector_stores(config):
    """
    Get image and text vector stores
    """

    fields = get_fields(config.AZURE_OPENAI_EMBEDDING_DIMENSIONS)
    my_embedding_function = MyAzureOpenAIEmbeddings(
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        dimensions=config.AZURE_OPENAI_EMBEDDING_DIMENSIONS,
    ).embed_query

    vector_search = get_vector_search(
        algorithm_configuration_name=config.ALGORITHM_CONFIGURATION_NAME,
        azure_openai_embedding_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        azure_openai_endpoint=config.AZURE_OPENAI_ENDPOINT,
        azure_openai_key=config.AZURE_OPENAI_API_KEY,
        azure_openai_model_name=config.AZURE_OPENAI_MODEL_NAME,
        vector_search_profile_name=config.VECTOR_SEARCH_PROFILE_NAME,
        vectorizer_name=config.VECTORIZER_NAME,
    )

    semantic_search = get_semantic_search(
        semantic_configuration_name=config.SEMANTIC_CONFIGURATION_NAME,
        field_name="chunk",
    )

    summary_vector_store = MyAzureSearch(
        azure_search_endpoint=config.AZURE_SEARCH_SERVICE_ENDPOINT,
        azure_search_key=config.AZURE_SEARCH_ADMIN_KEY,
        index_name=config.SUMMARY_INDEX_NAME,
        embedding_function=my_embedding_function,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    text_vector_store = MyAzureSearch(
        azure_search_endpoint=config.AZURE_SEARCH_SERVICE_ENDPOINT,
        azure_search_key=config.AZURE_SEARCH_ADMIN_KEY,
        index_name=config.TEXT_INDEX_NAME,
        embedding_function=my_embedding_function,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    image_vector_store = MyAzureSearch(
        azure_search_endpoint=config.AZURE_SEARCH_SERVICE_ENDPOINT,
        azure_search_key=config.AZURE_SEARCH_ADMIN_KEY,
        index_name=config.IMAGE_INDEX_NAME,
        embedding_function=my_embedding_function,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    return {
        "summary_vector_store": summary_vector_store,
        "text_vector_store": text_vector_store,
        "image_vector_store": image_vector_store,
    }
