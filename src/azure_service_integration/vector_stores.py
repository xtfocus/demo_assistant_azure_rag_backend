import json
from collections.abc import Callable
from typing import List, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (SearchIndex, SemanticSearch,
                                                   VectorSearch)
from loguru import logger
from openai import AzureOpenAI

from src.file_processing.models import BaseChunk, FileMetadata


class MyAzureSearch:
    """
    Represent an Azure AI Search Index
    """

    def __init__(
        self,
        azure_search_endpoint: str,
        azure_search_key: str,
        index_name: str,
        embedding_function: Callable,
        fields: List,
        vector_search: VectorSearch,
        semantic_search: SemanticSearch,
    ):
        self.endpoint = azure_search_endpoint
        self.index_name = index_name
        self.fields = fields
        self.embedding_function = embedding_function

        # Create clients for interacting with the search service and index
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(azure_search_key),
        )

        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(azure_search_key)
        )

        self.vector_search = vector_search
        self.semantic_search = semantic_search

        # Ensure the index exists or create it if not
        self._create_index_if_not_exists()

    def _create_index_if_not_exists(self) -> None:
        """Creates the index if it does not already exist."""
        try:
            # Check if the index exists
            self.index_client.get_index(name=self.index_name)
            logger.info(f"Index '{self.index_name}' already exists.")

        except ResourceNotFoundError:
            index = SearchIndex(
                name=self.index_name,
                fields=self.fields,
                vector_search=self.vector_search,
                semantic_search=self.semantic_search,
            )
            self.index_client.create_index(index)
            logger.info(f"Index '{self.index_name}' has been created.")

    async def upload_documents(self, documents) -> None:
        """Uploads documents to the Azure Search index."""
        self.search_client.upload_documents(documents=documents)

    @staticmethod
    def filtered_texts_and_metadatas_by_min_length(
        texts, metadatas, min_len=10
    ) -> Tuple:

        # Filter texts and metadatas where text length is >= 10
        filtered_batch = [
            (text, metadata)
            for text, metadata in zip(texts, metadatas)
            if len(text) >= min_len
        ]

        # Unpack the filtered batch back into separate lists
        filtered_texts, filtered_metadatas = (
            zip(*filtered_batch) if filtered_batch else ([], [])
        )

        diff = len(texts) - len(filtered_texts)

        if diff:
            logger.info(
                f"{diff} texts removed by length: {[i for i in texts if len(i) < min_len]}"
            )

        return filtered_texts, filtered_metadatas

    async def add_texts(
        self,
        texts: List[str],
        metadatas=None,
        batch_size: int = 10,
        filter_by_min_len: int = 0,
    ):
        """Adds texts and their associated metadata to the Azure Search index."""
        documents = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = (
                metadatas[i : i + batch_size] if metadatas else [{}] * len(batch_texts)
            )

            if filter_by_min_len:
                filtered_texts, filtered_metadatas = (
                    self.filtered_texts_and_metadatas_by_min_length(
                        batch_texts, batch_metadatas, min_len=filter_by_min_len
                    )
                )
            else:
                filtered_texts, filtered_metadatas = batch_texts, batch_metadatas

            if not bool(filtered_texts):
                continue

            try:
                # Batch embed texts
                embeddings = self.embedding_function(filtered_texts)
            except Exception as e:
                logger.error(f" Error during text embedding for batch {i}: {str(e)}")
                logger.error(
                    "Showing batch \n" + "<end>\n---\n<start>".join(filtered_texts)
                )
                raise

            for text, embedding, metadata in zip(
                filtered_texts, embeddings, filtered_metadatas
            ):
                doc = {
                    "chunk_id": metadata["chunk_id"],
                    "chunk": text or "no description",
                    "vector": embedding,
                    "metadata": json.dumps(metadata["metadata"]),
                    "parent_id": metadata["parent_id"],
                    "title": metadata["title"],
                    "uploader": metadata["uploader"],
                    "upload_time": metadata["upload_time"],
                }
                documents.append(doc)

        if documents:
            # Upload prepared documents to the index
            upload_success = await self.upload_documents(documents)
            return upload_success

    @staticmethod
    def create_texts_and_metadatas(
        chunks: List[BaseChunk], file_metadata: FileMetadata, prefix="text"
    ):
        """
        Given BaseChunk and Parent file metadata, prepare texts and metadata to
        be used with `add_texts`
        """
        # Extract texts and metadata
        texts = [chunk.chunk for chunk in chunks]
        metadatas = [
            {
                "chunk_id": f"{prefix}_{file_metadata['file_hash']}_{chunk.chunk_no}",
                "metadata": json.dumps({"page_range": chunk.page_range.dict()}),
                "title": file_metadata["title"],
                "parent_id": file_metadata["file_hash"],
                "uploader": file_metadata["uploader"],
                "upload_time": file_metadata.get("upload_time"),
            }
            for chunk in chunks
        ]

        return texts, metadatas


class MyAzureOpenAIEmbeddings:
    def __init__(self, api_key, api_version, azure_endpoint, model, dimensions):
        """
        Initializes the MyAzureOpenAIEmbeddings instance.

        Args:
            api_key (str): Azure OpenAI API key.
            api_version (str): Azure OpenAI API version.
            azure_endpoint (str): Azure OpenAI endpoint.
            model (str): The embedding model deployment name.
        """
        self.client = AzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
        )
        self.model = model
        self.dimensions = dimensions

    def embed_query(self, texts: List[str]) -> List[list]:
        """
        Generates embeddings for a batch of texts.

        Args:
            texts (List[str]): List of input texts to generate embeddings for.

        Returns:
            List[list]: List of embedding vectors.
        """
        response = self.client.embeddings.create(
            input=texts, model=self.model, dimensions=self.dimensions
        )
        return [item.embedding for item in response.data]
