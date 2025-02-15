import asyncio
import os
import uuid
from collections.abc import Iterable
from typing import List

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from loguru import logger

from src.configuration.globals import clients, objects
from src.helpers.delete_helpers import process_deletion_across_indices
from src.pipeline import MyFile, ProcessingResult

router = APIRouter()

# Shared results store (use a more robust storage mechanism in production)
background_results = {}


@router.post("/api/exec/uploads/")
async def process_uploaded_files(
    user_name: str,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Process multiple uploaded files asynchronously in the background.
    Args:
        files: List of uploaded files from the client
        user_name: user_name
        background_tasks: FastAPI BackgroundTasks object
    Returns:
        Dictionary with task_id for tracking the background processing
    """
    task_id = str(uuid.uuid4())
    background_results[task_id] = {"status": "processing", "results": []}

    # Read all file contents before starting background task
    file_data = []
    for file in files:
        try:
            content = await file.read()
            file_data.append({"filename": file.filename, "content": content})
        except Exception as e:
            logger.error(f"Error reading file '{file.filename}': {str(e)}")
            background_results[task_id]["results"].append(
                {"file_name": file.filename, "error": f"Error reading file: {str(e)}"}
            )
            continue

    async def process_files_in_background():
        pipeline = objects["pipeline"]
        objects["duplicate-checker"]._ensure_container_exists()
        results = []

        for file_info in file_data:
            try:
                file_marked_name = f"{user_name}_{file_info['filename']}"

                if not objects["duplicate-checker"].duplicate_by_file_name(
                    file_marked_name
                ):
                    my_file = MyFile(
                        file_name=file_info["filename"],
                        file_content=file_info["content"],
                        uploader=user_name,
                    )

                    result: ProcessingResult = await pipeline.process_file(my_file)
                    if not result.errors:
                        objects["duplicate-checker"].update(file_name=file_marked_name)

                    results.append(
                        {"file_name": file_info["filename"], "result": result}
                    )
                else:
                    logger.warning(
                        f"File {file_marked_name} already processed. Skipping"
                    )
                    results.append(
                        {
                            "file_name": file_info["filename"],
                            "error": "File already processed",
                        }
                    )

            except Exception as e:
                logger.error(
                    f"Error processing file '{file_info['filename']}': {str(e)}"
                )
                results.append({"file_name": file_info["filename"], "error": str(e)})

        objects["duplicate-checker"].save()
        background_results[task_id]["status"] = "completed"
        background_results[task_id]["results"] = results

    background_tasks.add_task(process_files_in_background)

    return {
        "task_id": task_id,
        "message": "Files are being processed in the background",
        "status_endpoint": f"/api/exec/upload-status/{task_id}",
    }


@router.get("/api/exec/upload-status/{task_id}")
async def get_upload_status(task_id: str):
    """
    Get the status and results of a background file processing task.
    Args:
        task_id: UUID of the background task
    Returns:
        Dictionary containing status and results of the processing task
    """
    if task_id not in background_results:
        raise HTTPException(status_code=404, detail="Task not found")

    return background_results[task_id]


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
    duplicate_checker = objects["duplicate-checker"]

    # First identify all files to be removed
    files_to_remove = [
        file_name
        for file_name in duplicate_checker.known_dict["known_file_names"]
        if file_name.startswith(f"{user_name}_")
    ]

    try:
        # Remove all marked file names for this user from cache
        removed_file_names = []

        for file_name in files_to_remove:
            if file_name.startswith(f"{user_name}_"):
                if duplicate_checker.remove_file_name(file_name):
                    removed_file_names.append(
                        file_name
                    )  # First handle the image files for the image search client

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
            "removed_cache_entries": {
                "count": len(removed_file_names),
                "files": removed_file_names,
            },
        }

    except Exception as e:
        error_msg = f"Error removing user data for '{user_name}': {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
