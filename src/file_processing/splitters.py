"""
File        : splitters.py
Author      : tungnx23
Description : Simplified splitter function that work with list of texts 
where position information must be preserved
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from src.file_processing.models import BaseChunk, PageRange


class BaseTextSplitter(ABC):
    """
    Abstract base class for text splitters. Defines the interface for
    splitting text into chunks with metadata.
    """

    @abstractmethod
    def split_text(self, pages: List[dict]) -> List[BaseChunk]:
        """
        Splits text from pages into chunks with metadata.

        Args:
            pages: List of pages, where each page contains page_no and text fields.

        Returns:
            List of BaseChunk objects containing split text and metadata.
        """
        pass

    @staticmethod
    def _create_chunk(
        chunk: str, chunk_no: str, page_range: tuple[int, int]
    ) -> BaseChunk:
        """
        Create a new chunk object with metadata.

        Args:
            current_chunk: Text content for the chunk.
            chunk_no: Index number for this chunk.
            page_range: Tuple of (start_page, end_page) for this chunk.

        Returns:
            New BaseChunk object with the provided data.
        """
        return BaseChunk(
            chunk_no=chunk_no,
            page_range=PageRange(start_page=page_range[0], end_page=page_range[1]),
            chunk=chunk,
        )


class SimplePageTextSplitter(BaseTextSplitter):
    """
    A text splitter that chunks text from pages while maintaining page number.
    Probably doesn't work with other length function other than len
    """

    DEFAULT_SEPARATORS = ["\n\n", ".\n", ". ", "\n", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        length_function: Callable[[str], int] = len,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the text splitter with configurable parameters.

        Args:
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Overlap size between consecutive chunks.
            length_function: Function to calculate text length.
            separators: List of separators to use for splitting.

        Raises:
            ValueError: If chunk_size or chunk_overlap are invalid.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._separators = separators or self.DEFAULT_SEPARATORS

    def split_text(self, pages: List[dict]) -> List[BaseChunk]:
        """
        Splits text from pages into chunks, ensuring each chunk respects the
        chunk size and ends at one of the defined separators.

        Args:
            pages: List of pages, where each page contains page_no and text fields.

        Returns:
            List of BaseChunk objects containing the split text and metadata.

        Raises:
            ValueError: If input pages are invalid or empty.
        """
        if not pages:
            raise ValueError("Input pages list cannot be empty")

        chunks: List[BaseChunk] = []
        current_chunk = ""
        overlap_text = ""
        current_page_range = (0, 0)

        for page in pages:
            page_no, page_text = page["page_no"], page["text"]
            remaining_text = page_text

            if not remaining_text:
                continue

            while remaining_text:
                # Update page range
                current_page_range = (
                    (current_page_range[0] - 1) if current_chunk else page_no,
                    page_no,
                )

                # Check if remaining text fits in current chunk
                if (
                    self._length_function(current_chunk + remaining_text)
                    <= self._chunk_size
                ):
                    current_chunk += remaining_text
                    remaining_text = ""
                else:
                    split_point = self._find_split_point(remaining_text)
                    current_chunk += remaining_text[:split_point]
                    remaining_text = remaining_text[split_point:].lstrip()

                # Check if chunk is ready to be added
                current_length = self._length_function(current_chunk)
                overlap_length = self._length_function(overlap_text)

                if current_length + overlap_length >= self._chunk_size:
                    current_chunk = overlap_text + current_chunk
                    chunk = self._create_chunk(
                        chunk=current_chunk,
                        chunk_no=str(len(chunks)),
                        page_range=current_page_range,
                    )
                    chunks.append(chunk)

                    overlap_text = self._create_overlap_text(current_chunk)
                    current_chunk = ""

        # Handle remaining text
        if current_chunk:
            chunk = self._create_chunk(
                chunk=overlap_text + current_chunk,
                chunk_no=str(len(chunks)),
                page_range=current_page_range,
            )
            chunks.append(chunk)

        return chunks

    def _find_split_point(self, text: str) -> int:
        """
        Find the optimal split point in the text using separators.

        Args:
            text: Text to split.

        Returns:
            Index where the text should be split.
        """
        for separator in self._separators:
            split_idx = text.rfind(separator, 0, self._chunk_size)
            if split_idx != -1:
                return split_idx + len(separator)

        return min(self._chunk_size, len(text))

    def _create_overlap_text(self, chunk: str) -> str:
        """
        Create overlap text for the next chunk based on separators.

        Args:
            chunk: Current chunk to create overlap from.

        Returns:
            Text to be used as overlap in the next chunk.
        """
        if self._chunk_overlap == 0 or len(chunk) <= self._chunk_overlap:
            return chunk

        min_overlap_start = max(0, len(chunk) - self._chunk_overlap)

        # Try to expand left to find a clean separator
        for separator in self._separators:
            split_idx = chunk.rfind(separator, 0, min_overlap_start)
            if split_idx != -1:
                return chunk[split_idx + len(separator) :]

        # If no separator found, use exact overlap size
        return chunk[min_overlap_start:]
