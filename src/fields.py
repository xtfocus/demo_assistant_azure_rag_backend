from dataclasses import field

from azure.search.documents.indexes.models import (SearchableField,
                                                   SearchField,
                                                   SearchFieldDataType,
                                                   SimpleField)


def get_fields(azure_openai_embedding_dimensions: int):
    fields = [
        SearchField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            analyzer_name="keyword",
            facetable=True,  # Can be used for faceted navigation
            filterable=True,  # Can be used in filter expressions
            key=True,  # Primary key field for the document
            sortable=True,  # Results can be sorted by this field
        ),
        SearchableField(
            # This field stores the actual text content of each chunk and
            # is optimized for full-text search.
            name="chunk",
            type=SearchFieldDataType.String,
            facetable=False,  # Not used for faceting
            filterable=False,  # Not used for filtering
            searchable=True,  # Full-text search enabled
            sortable=False,  # Cannot sort results by this field
        ),
        SearchField(
            # This field stores the vector embeddings for semantic search,
            # using the HNSW (Hierarchical Navigable Small World) algorithm
            # for efficient similarity search.
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=azure_openai_embedding_dimensions,
            vector_search_profile_name="myHnswProfile",
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        # Additional field to store the title
        SearchableField(
            # Stores the title of the document with full-text search capability
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
        ),
        # Additional field for filtering on document source
        SimpleField(
            # Used to link chunks to their parent document, enabling
            # filtering by source document.
            name="parent_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
    ]
    return fields
