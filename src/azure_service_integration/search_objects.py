"""
File        : azure_vector_search.py
Author      : tungnx23
Description : Helper function to create a Azure VectorSearch and SemanticSearch objects. 
"""

from azure.search.documents.indexes.models import (AzureOpenAIParameters,
                                                   AzureOpenAIVectorizer,
                                                   HnswAlgorithmConfiguration,
                                                   SemanticConfiguration,
                                                   SemanticField,
                                                   SemanticPrioritizedFields,
                                                   SemanticSearch,
                                                   VectorSearch,
                                                   VectorSearchProfile)


def get_vector_search(
    algorithm_configuration_name: str,
    azure_openai_embedding_deployment: str,
    azure_openai_endpoint: str,
    azure_openai_key: str,
    azure_openai_model_name: str,
    vector_search_profile_name: str,
    vectorizer_name: str,
) -> VectorSearch:

    # Vector search configuration
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name=algorithm_configuration_name),
        ],
        profiles=[
            VectorSearchProfile(
                name=vector_search_profile_name,
                algorithm_configuration_name=algorithm_configuration_name,
                vectorizer=vectorizer_name,
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                name=vectorizer_name,
                kind="azureOpenAI",
                azure_open_ai_parameters=AzureOpenAIParameters(
                    resource_uri=azure_openai_endpoint,
                    deployment_id=azure_openai_embedding_deployment,
                    model_name=azure_openai_model_name,
                    api_key=azure_openai_key,
                ),
            ),
        ],
    )
    return vector_search


def get_semantic_search(
    semantic_configuration_name: str, field_name: str = "chunk"
) -> SemanticSearch:
    semantic_config = SemanticConfiguration(
        name=semantic_configuration_name,
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name=field_name)]
        ),
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])
    return semantic_search
