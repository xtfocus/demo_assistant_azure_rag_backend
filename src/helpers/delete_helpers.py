import asyncio
from typing import Dict, List

from azure.search.documents import SearchClient
from loguru import logger


async def delete_image_files(chunk_ids: List[str], image_container_client) -> List[str]:
    """
    Delete image files from blob storage based on chunk IDs.

    Args:
        chunk_ids: List of chunk IDs corresponding to image files
        image_container_client: Azure container client for image storage

    Returns:
        List of successfully deleted blob names
    """

    deleted_blobs = []

    for chunk_id in chunk_ids:
        blob_name = f"{chunk_id}"
        if await asyncio.to_thread(image_container_client.delete_file, blob_name):
            deleted_blobs.append(blob_name)
        else:
            logger.warning(f"Failed to delete image file: {blob_name}")

    logger.info(
        f"Deleted {len(deleted_blobs)} image files out of {len(chunk_ids)} found"
    )

    return deleted_blobs


async def delete_documents_from_search(
    search_client: SearchClient, filter_expression: str, batch_size: int = 1000
) -> dict:
    """
    Delete documents from Azure Search based on a filter expression.

    Args:
        search_client: Azure Search client
        filter_expression: OData filter expression
        batch_size: Size of deletion batches (default 1000)

    Returns:
        Dictionary containing deletion results
    """

    try:
        # Search for documents matching the filter
        search_results = await asyncio.to_thread(
            search_client.search,
            search_text="*",
            filter=filter_expression,
            select=["chunk_id"],
        )

        # Collect chunk IDs
        chunk_ids = [result["chunk_id"] for result in search_results]

        if not chunk_ids:
            return {
                "index": search_client._index_name,
                "status": "no_documents_found",
                "documents_removed": 0,
                "chunk_ids": [],
            }

        # Delete documents in batches
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i : i + batch_size]
            await asyncio.to_thread(
                search_client.delete_documents,
                documents=[
                    {"@search.action": "delete", "chunk_id": chunk_id}
                    for chunk_id in batch
                ],
            )

        return {
            "index": search_client._index_name,
            "status": "success",
            "documents_removed": len(chunk_ids),
            "chunk_ids": chunk_ids,
        }

    except Exception as e:
        error_msg = (
            f"Error removing documents from {search_client._index_name}: {str(e)}"
        )
        logger.error(error_msg)

        return {
            "index": search_client._index_name,
            "status": "error",
            "error": str(e),
            "documents_removed": 0,
            "chunk_ids": [],
        }


async def process_deletion_across_indices(
    search_clients: Dict[str, SearchClient],
    filter_expression: str,
    image_container_client=None,
) -> dict:
    """
    Process deletion across multiple search indices and handle image deletion.

    Args:
        search_clients: List of search clients to process
        filter_expression: OData filter expression for deletion
        image_container_client: Optional client for image deletion

    Returns:
        Dictionary containing overall deletion results
    """

    results = []
    total_removed = 0
    deleted_blobs = []

    # First handle image search and file deletion if needed
    if image_container_client:
        image_search_result = await delete_documents_from_search(
            search_clients["image-azure-ai-search"], filter_expression
        )

        if image_search_result["status"] == "success":
            deleted_blobs = await delete_image_files(
                image_search_result["chunk_ids"], image_container_client
            )

    # Process each search client
    for client in search_clients.values():
        result = await delete_documents_from_search(client, filter_expression)
        results.append(result)
        total_removed += result["documents_removed"]

    return {
        "overall_status": "completed",
        "total_documents_removed": total_removed,
        "index_results": results,
        "deleted_image_files": {"count": len(deleted_blobs), "files": deleted_blobs},
    }
