import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field


class CustomSkillException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        logger.error(
            f"CustomSkillException raised: {self.message}"
        )  # Log exception on creation
        super().__init__(self.message)


class RequestData(BaseModel):
    values: List[Dict]


class PageRange(BaseModel):
    """Represents the page range information for a document chunk"""

    start_page: int
    end_page: int


class BaseChunk(BaseModel):
    """Represents a single chunk of text with its metadata"""

    chunk_no: str
    chunk: str
    page_range: Optional[PageRange]


class FileMetadata(BaseModel):
    """Represents the metadata for a file"""

    file_hash: str
    title: str
    created_at: datetime = Field(default_factory=datetime.now)
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)


class AzureSearchDoc(BaseModel):
    """
    Represents a document in Azure Search index with all required fields
    """

    chunk_id: str = Field(description="Unique identifier for the chunk")
    chunk: str = Field(description="The actual text content")
    vector: List[float] = Field(description="Vector embedding of the chunk")
    metadata: str = Field(description="JSON serialized metadata")
    parent_id: str = Field(description="ID of the parent document")
    title: str = Field(description="Title of the document")

    @classmethod
    def from_chunk(
        cls,
        chunk: BaseChunk,
        file_metadata: FileMetadata,
        embedding_function: Callable,
    ) -> "AzureSearchDoc":
        """
        Creates an AzureSearchDoc from a chunk and file metadata
        """
        return cls(
            chunk_id=f"{file_metadata.file_hash}_chunk_{chunk.chunk_no}",
            chunk=chunk.chunk,
            vector=embedding_function(chunk.chunk),
            metadata=json.dumps({"page_range": chunk.page_range.dict()}),
            parent_id=file_metadata.file_hash,
            title=file_metadata.title,
        )
