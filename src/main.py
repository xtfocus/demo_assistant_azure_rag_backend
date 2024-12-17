import asyncio
import os
from collections.abc import Iterable
from typing import Any, Callable, List

from azure.search.documents import SearchClient
from fastapi import (APIRouter, BackgroundTasks, Depends, File, HTTPException,
                     UploadFile)
from loguru import logger

from src import task_counter
from src.delete_helpers import process_deletion_across_indices
from src.pipeline import MyFile, ProcessingResult
from src.task_counter import TaskCounter

from .globals import clients, objects

router = APIRouter()

# Shared results store (use a more robust storage mechanism in production)
background_results = {}

task_counter = TaskCounter()


async def ensure_no_active_tasks():
    """
    Dependency that checks if there are any active background tasks.
    """

    if task_counter.is_busy:
        raise HTTPException(
            status_code=409,
            detail=f"There are {task_counter.active_tasks} background tasks still running. Please try again later.",
        )
    yield


def run_with_task_counter(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to wrap a function with task counter increment and decrement logic.
    """

    async def wrapper(*args, **kwargs):
        task_counter.increment()
        try:
            return await func(*args, **kwargs)
        finally:
            task_counter.decrement()

    return wrapper


@router.post("/api/exec/uploads/")
async def process_uploaded_files(
    user_name: str,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    _: None = Depends(ensure_no_active_tasks),
):
    """
    Process multiple uploaded files asynchronously.
    Args:
        files: List of uploaded files from the client.
        user_name: user_name
    Returns:
        List of results for each processed file.
    """

    pipeline = objects["pipeline"]
    objects["duplicate-checker"]._ensure_container_exists()

    @run_with_task_counter
    async def process_single_file(file: UploadFile):
        try:
            # Read file content asynchronously
            file_content = await file.read()
            file_marked_name = (
                user_name + "_" + file.filename
            )  # marking file name  with user name, to be stored in cache
            if not objects["duplicate-checker"].duplicate_by_file_name(
                file_marked_name
            ):
                my_file = MyFile(
                    file_name=file.filename,
                    file_content=file_content,
                    uploader=user_name,
                )

                # Process the file using the pipeline
                result: ProcessingResult = await pipeline.process_file(my_file)
                if not (result.errors):
                    objects["duplicate-checker"].update(file_name=file_marked_name)

                return {"file_name": file.filename, "result": result}
            else:
                logger.warning(f"File {file_marked_name} already processed. Skipping")
                raise ValueError(f"{file.filename} already processed. Skipping...")

        except Exception as e:
            # Raise HTTPException for any errors
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file '{file.filename}': {str(e)}",
            )

    # Create tasks for processing each file
    tasks = [process_single_file(file) for file in files]

    # Gather results for all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle results and exceptions
    final_results = []
    for index, result in enumerate(results):
        if isinstance(result, Exception):
            # Log or return the error for this file
            final_results.append(
                {"file_name": files[index].filename, "error": str(result)}
            )
        else:
            final_results.append(result)

    objects["duplicate-checker"].save()
    return final_results


async def search_client_filter_file(file_name: str, search_client) -> Iterable:
    """ """
    # Get file name without extension for title matching
    title = os.path.splitext(file_name)[0]
    filter_expr = f"title eq '{title}'"
    search_results = await asyncio.to_thread(
        search_client.search,
        search_text="*",  # Get all documents
        filter=filter_expr,  # Exact match using OData filter
        select=["chunk_id", "chunk", "metadata"],  # Only get chunk_ids for efficiency
    )

    return list(search_results)


async def remove_file(file_name: str, search_client) -> dict:
    """
    Remove all documents from Azure Search where either:
    - title exactly matches the file name (without extension), or

    Args:
        file_name: Name of the file to remove (with extension)
        search_client: Azure Search client instance

    Returns:
        dict: Result of the removal operation including number of documents removed
    """
    try:
        # Get file name without extension for title matching
        title = os.path.splitext(file_name)[0]
        filter_expr = f"title eq '{title}'"
        search_term = title

        # Search for documents with exact match using OData filter
        search_results = await asyncio.to_thread(
            search_client.search,
            search_text="*",  # Get all documents
            filter=filter_expr,  # Exact match using OData filter
            select=["chunk_id"],  # Only get chunk_ids for efficiency
        )

        # Collect all chunk_ids
        chunk_ids = []
        for result in search_results:
            chunk_ids.append(result["chunk_id"])

        if not chunk_ids:
            field_type = "title"
            logger.warning(f"No documents found with {field_type} '{search_term}'")
            return {
                "file_name": file_name,
                "status": "no_documents_found",
                "documents_removed": 0,
                "filter_type": field_type,
            }

        # Delete documents in batches
        batch_size = 1000  # Azure Search limitation
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i : i + batch_size]
            await asyncio.to_thread(
                search_client.delete_documents,
                documents=[
                    {"@search.action": "delete", "chunk_id": chunk_id}
                    for chunk_id in batch
                ],
            )

        field_type = "title"
        logger.info(
            f"Successfully removed {len(chunk_ids)} documents for file '{file_name}' using {field_type} filter"
        )
        return {
            "file_name": file_name,
            "status": "success",
            "documents_removed": len(chunk_ids),
            "filter_type": field_type,
        }

    except Exception as e:
        error_msg = f"Error removing documents for file '{file_name}': {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.delete("/api/exec/remove_file/")
async def remove_file_endpoint(
    user_name: str,
    file_name: str,
    _: None = Depends(ensure_no_active_tasks),
):
    """
    Remove all documents associated with a file from multiple Azure Search clients
    and delete associated image files if present.
    """
    title = os.path.splitext(file_name)[0]
    marked_file_name = f"{user_name}_{file_name}"  # Create marked file name

    try:
        result = await process_deletion_across_indices(
            search_clients={
                name: clients[name]
                for name in [
                    "text-azure-ai-search",
                    "image-azure-ai-search",
                    "summary-azure-ai-search",
                ]
            },
            filter_expression=f"title eq '{title}'",
            image_container_client=clients["image_container_client"],
        )

        # Remove the marked file name from cache
        duplicate_checker = objects["duplicate-checker"]
        if duplicate_checker.remove_file_name(marked_file_name):
            logger.info(f"Removed marked file name {marked_file_name} from cache")

        return {"file_name": file_name, **result}

    except Exception as e:
        error_msg = f"Error removing file data for '{file_name}': {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.delete("/api/exec/remove_user_data/")
async def remove_user_data_endpoint(
    user_name: str,
    _: None = Depends(ensure_no_active_tasks),
):
    """

    Remove all documents and associated images belonging to a specific user.


    Args:

        user_name: Name of the user whose data should be removed


    Returns:

        Result of the removal operation from all search clients and blob storage

    """

    results = []

    total_removed = 0

    deleted_blobs = []

    try:

        # First handle the image files for the image search client

        image_container_client = clients["image_container_client"]

        # Search for all documents with matching uploader in image index

        search_results = await asyncio.to_thread(
            clients["image-azure-ai-search"].search,
            search_text="*",
            filter=f"uploader eq '{user_name}'",
            select=["chunk_id"],
        )

        # Delete all associated image files first
        chunk_ids = []

        for doc in search_results:
            chunk_ids.append(doc["chunk_id"])
            blob_name = f"{doc['chunk_id']}"  # Assuming blob name matches chunk_id

            if await asyncio.to_thread(image_container_client.delete_file, blob_name):
                deleted_blobs.append(blob_name)
            else:
                logger.warning(f"Failed to delete image file: {blob_name}")

        logger.info(
            f"Deleted {len(deleted_blobs)} image files out of {len(chunk_ids)} found"
        )

        # Then process each search client to delete documents

        for client in [
            clients["text-azure-ai-search"],
            clients["image-azure-ai-search"],
            clients["summary-azure-ai-search"],
        ]:

            try:
                # Search for documents with matching uploader
                search_results = await asyncio.to_thread(
                    client.search,
                    search_text="*",
                    filter=f"uploader eq '{user_name}'",
                    select=["chunk_id"],
                )

                # Collect all chunk_ids
                chunk_ids = [result["chunk_id"] for result in search_results]

                if not chunk_ids:
                    logger.warning(
                        f"No documents found for user '{user_name}' in {client._index_name}"
                    )

                    results.append(
                        {
                            "index": client._index_name,
                            "status": "no_documents_found",
                            "documents_removed": 0,
                        }
                    )
                    continue

                # Delete documents in batches

                batch_size = 1000  # Azure Search limitation

                for i in range(0, len(chunk_ids), batch_size):
                    batch = chunk_ids[i : i + batch_size]
                    await asyncio.to_thread(
                        client.delete_documents,
                        documents=[
                            {"@search.action": "delete", "chunk_id": chunk_id}
                            for chunk_id in batch
                        ],
                    )

                logger.info(
                    f"Successfully removed {len(chunk_ids)} documents for user '{user_name}' from {client._index_name}"
                )

                results.append(
                    {
                        "index": client._index_name,
                        "status": "success",
                        "documents_removed": len(chunk_ids),
                    }
                )

                total_removed += len(chunk_ids)

            except Exception as e:
                error_msg = (
                    f"Error removing documents from {client._index_name}: {str(e)}"
                )
                logger.error(error_msg)
                results.append(
                    {
                        "index": client._index_name,
                        "status": "error",
                        "error": str(e),
                        "documents_removed": 0,
                    }
                )

        return {
            "user_name": user_name,
            "overall_status": "completed",
            "total_documents_removed": total_removed,
            "index_results": results,
            "deleted_image_files": {
                "count": len(deleted_blobs),
                "files": deleted_blobs,
            },
        }

    except Exception as e:
        error_msg = f"Error removing user data for '{user_name}': {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
