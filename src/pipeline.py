import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from src.azure_service_integration.azure_container_client import \
    AzureContainerClient
from src.azure_service_integration.vector_stores import MyAzureSearch
from src.file_processing.file_summarizer import FileSummarizer
from src.file_processing.file_utils import (create_file_metadata_from_bytes,
                                            pdf_blob_to_pymupdf_doc)
from src.file_processing.image_descriptor import ImageDescriptor
from src.file_processing.pdf_parsing import FileImage, extract_texts_and_images
from src.file_processing.splitters import SimplePageTextSplitter
from src.models import BaseChunk, PageRange


class MyFile(BaseModel):
    file_name: str
    file_content: bytes
    uploader: str = "default"
    upload_time: datetime = Field(default_factory=datetime.now)


class ProcessingResult(NamedTuple):
    """Structured return type for process_file method"""

    file_name: str
    num_pages: int
    num_texts: int
    num_images: int
    metadata: Dict[str, Any]
    errors: Optional[List[str]] = None


class Pipeline:
    """
    Orchestrating the extracting > chunking > embedding > indexing using Azure resources
    """

    def __init__(
        self,
        text_vector_store: MyAzureSearch,
        image_vector_store: MyAzureSearch,
        summary_vector_store: MyAzureSearch,
        embedding_function: Callable,
        text_splitter: SimplePageTextSplitter,
        image_descriptor: ImageDescriptor,
        file_summarizer: FileSummarizer,
        image_container_client: AzureContainerClient,
    ):
        """Initialize the pipeline with necessary components

        Args:
            text_vector_store: Vector store for text chunks
            image_vector_store: Vector store for image descriptions
            embedding_function: Function to create embeddings
            text_splitter: Text splitting strategy
            image_descriptor: OpenAI client wrapper for image description
            image_container_client: client wrapper for image storage
        """
        self.text_vector_store = text_vector_store
        self.image_vector_store = image_vector_store
        self.summary_vector_store = summary_vector_store
        self.embedding_function = embedding_function
        self.text_splitter = text_splitter
        self.image_descriptor = image_descriptor
        self.file_summarizer = file_summarizer
        self.image_container_client = image_container_client

    async def _process_images(
        self, images: List[FileImage], summary, max_concurrent_requests: int = 30
    ) -> List[str]:
        """Process multiple images concurrently with rate limiting using a semaphore."""
        semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def process_single_image(image):
            async with semaphore:
                return await self.image_descriptor.run(image.image_base64, summary)

        tasks = [process_single_image(img) for img in images]
        return await asyncio.gather(*tasks)

    def _create_text_chunks(
        self, texts: List[Any], file_metadata: Dict
    ) -> Tuple[List[str], List[Dict]]:
        """Create text chunks and their metadata

        Args:
            texts: List of text objects
            file_metadata: Metadata about the file

        Returns:
            Tuple containing lists of texts and their metadata
        """
        text_chunks = self.text_splitter.split_text((text.dict() for text in texts))
        return self.text_vector_store.create_texts_and_metadatas(
            text_chunks, file_metadata, prefix="text"
        )

    def _create_image_chunks(
        self, images: List[Any], descriptions: List[str], file_metadata: Dict
    ) -> Tuple[List[str], List[Dict]]:
        """Create image chunks and their metadata

        Args:
            images: List of image objects
            descriptions: List of image descriptions
            file_metadata: Metadata about the file

        Returns:
            Tuple containing lists of image texts and their metadata
        """
        image_chunks = [
            BaseChunk(
                chunk_no=f"{img.page_no}_{img.image_no}",
                page_range=PageRange(start_page=img.page_no, end_page=img.page_no),
                chunk=desc,
            )
            for img, desc in zip(images, descriptions)
        ]
        return self.image_vector_store.create_texts_and_metadatas(
            image_chunks, file_metadata, prefix="image"
        )

    async def _create_and_add_text_chunks(self, texts: List[Any], file_metadata: Dict):
        """Combine creation and adding of text chunks"""
        if not texts:
            return None

        input_texts, input_metadatas = self._create_text_chunks(texts, file_metadata)
        return await self.text_vector_store.add_texts(
            texts=input_texts, metadatas=input_metadatas
        )

    async def _create_and_add_image_chunks(
        self, images: List[Any], descriptions: List[str], file_metadata: Dict
    ):
        """Combine creation and adding of image chunks"""
        if not images:
            return None

        image_texts, image_metadatas = self._create_image_chunks(
            images, descriptions, file_metadata
        )
        return (
            await self.image_vector_store.add_texts(
                texts=image_texts,
                metadatas=image_metadatas,
                filter_by_min_len=10,
            ),
            image_metadatas,
        )

    async def _create_summary(self, texts: List[str], images: List[FileImage]) -> str:
        """Just create the summary"""

        return await self.file_summarizer.run(texts, images)

    async def _add_file_summary_to_store(self, summary: str, file_metadata: Dict):
        """Add the summary to vector store"""
        summary_texts, summary_metadatas = (
            self.summary_vector_store.create_texts_and_metadatas(
                [
                    BaseChunk(
                        chunk=summary,
                        chunk_no="0",
                        page_range=PageRange(start_page=0, end_page=0),
                    )
                ],
                file_metadata,
                prefix="summary",
            )
        )

        return await self.summary_vector_store.add_texts(
            texts=summary_texts, metadatas=summary_metadatas
        )

    async def process_file(self, file: MyFile) -> ProcessingResult:
        """Process a single file through the pipeline with optimized concurrent operations"""

        errors = []
        file_name = file.file_name
        try:
            # Convert PDF to document
            with pdf_blob_to_pymupdf_doc(file.file_content) as doc:
                # Create file metadata
                file_metadata = create_file_metadata_from_bytes(
                    file_bytes=file.file_content, file_name=file.file_name
                )
                file_metadata["uploader"] = file.uploader
                file_metadata["upload_time"] = file.upload_time.isoformat()

                num_pages = len(doc)
                texts, images = extract_texts_and_images(doc, report=True)
                logger.info("Extracted raw texts and images")

            summary = ""
            # Create tasks dict to track all async operations
            tasks = {}
            # Start summary generation if we have content
            if texts or images:
                tasks["summary"] = asyncio.create_task(
                    self._create_summary([i.text for i in texts], images)
                )
            # Process texts if available
            if texts:
                tasks["text"] = asyncio.create_task(
                    self._create_and_add_text_chunks(texts, file_metadata)
                )

            # Wait for summary before processing images
            try:
                if "summary" in tasks:
                    summary = await tasks["summary"]
                    logger.info(f"Created summary for {file_name}")
                    tasks["summary_upload"] = asyncio.create_task(
                        self._add_file_summary_to_store(summary, file_metadata)
                    )
            except Exception as e:
                logger.error(f"Summary generation failed: {str(e)}")
                errors.append(f"Summary generation failed: {str(e)}")

            # Process images if available
            if images:
                try:
                    descriptions: List[str] = await self._process_images(
                        images,
                        summary=summary,
                    )

                    logger.info(f"Created image descriptions for {file_name}")

                    image_result, image_metadatas = (
                        await self._create_and_add_image_chunks(
                            images, descriptions, file_metadata
                        )
                    )

                    logger.info(f"Created image index for {file_name}")

                    tasks["image_upload"] = asyncio.create_task(
                        self.image_container_client.upload_base64_image_to_blob(
                            (i["chunk_id"] for i in image_metadatas),
                            (image.image_base64 for image in images),
                        )
                    )

                    logger.info(f"Saved images in {file_name}")
                except Exception as e:
                    logger.error(f"Image processing failed: {str(e)}")
                    errors.append(f"Image processing failed: {str(e)}")
                    raise
            # Wait for all remaining tasks to complete
            try:
                await asyncio.gather(*tasks.values())
            except Exception as e:
                logger.error(f"Task completion error: {str(e)}")
                errors.append(f"Task completion error: {str(e)}")
                raise

            logger.info(f"Processed file {file_name}")

            return ProcessingResult(
                file_name=file_name,
                num_pages=num_pages,
                num_texts=len(texts),
                num_images=len(images),
                metadata=file_metadata,
                errors=errors if errors else None,
            )
        except Exception as e:
            logger.error(f"Fatal error processing {file_name}: {str(e)}")
            return ProcessingResult(
                file_name=file_name,
                num_pages=0,
                num_texts=0,
                num_images=0,
                metadata={},
                errors=[f"Fatal error: {str(e)}"],
            )
