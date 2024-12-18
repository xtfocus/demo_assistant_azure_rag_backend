"""
File: azure_container_client.py
Desc: handling I/O tasks with Blob Storage for a specfic container

"""

import base64
from abc import ABC
from typing import Iterable, List, Optional

from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient
from loguru import logger


class BaseAzureContainerClient(ABC):
    """
    Abstract base class defining interface for Azure Blob Storage container operations
    """

    def __init__(
        self,
        client: BlobServiceClient,
        container_name: str = "default_container",
    ):
        """
        Initialize the base Azure container client.

        Args:
            client (BlobServiceClient): Azure Blob Storage service client
            container_name (str): Name of the container to manage
        """
        self.client: BlobServiceClient = client
        self.container_name: str = container_name
        logger.info(f"Making sure container {container_name} exists ...")
        self._ensure_container_exists()

    def list_blob_names(self) -> List[str]:
        return list(
            self.client.get_container_client(self.container_name).list_blob_names()
        )

    def _ensure_container_exists(self) -> None:
        """Check if the container exists and create it if not."""
        container_client: ContainerClient = self.client.get_container_client(
            self.container_name
        )
        logger.info(f"Check on {self.container_name}")
        if not container_client.exists():
            container_client.create_container()
            logger.info(f"Container '{self.container_name}' created.")
        else:
            logger.info(f"Container '{self.container_name}' already exists.")

    def download_file(self, blob_name: str) -> Optional[bytes]:
        """Download a file from the container.

        Args:
            blob_name (str): The name of the blob to download.

        Returns:
            Optional[bytes]: The content of the blob if found, otherwise None.
        """
        try:
            container_client: ContainerClient = self.client.get_container_client(
                self.container_name
            )
            blob_client: BlobClient = container_client.get_blob_client(blob_name)
            result = blob_client.download_blob().readall()
            logger.info(f"Successfully downloaded blob {blob_name}")
            return result
        except Exception as e:
            logger.error(f"Error downloading blob '{blob_name}': {e}")
            return None

    # Add this method to BaseAzureContainerClient class
    def delete_file(self, blob_name: str) -> bool:
        """
        Delete a file from the container.

        Args:
            blob_name (str): The name of the blob to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            container_client: ContainerClient = self.client.get_container_client(
                self.container_name
            )
            blob_client: BlobClient = container_client.get_blob_client(blob_name)

            if blob_client.exists():
                blob_client.delete_blob()
                logger.info(f"Successfully deleted blob {blob_name}")
                return True
            else:
                logger.warning(f"Blob {blob_name} does not exist")
                return False

        except Exception as e:
            logger.error(f"Error deleting blob '{blob_name}': {e}")
            return False


class AzureContainerClient(BaseAzureContainerClient):
    """
    Wrapper for BlobServiceClient to work on a single container
    """

    def __init__(
        self,
        client: BlobServiceClient,
        container_name: str = "default_container",
    ):
        """
        Initialize the Azure container client with a specified container name.

        Args:
            container_name (str): Name of the container to manage. Defaults to "default_container".
        """
        self.client: BlobServiceClient = client
        self.container_name: str = container_name
        logger.info(f"Making sure container {container_name} exists ...")
        self._ensure_container_exists()

    def list_pdf_files(self) -> List[str]:
        """List all PDF files in the container."""
        container_client: ContainerClient = self.client.get_container_client(
            self.container_name
        )
        return [
            blob.name
            for blob in container_client.list_blobs()
            if blob.name.endswith(".pdf")
        ]

    async def upload_base64_image_to_blob(
        self, blob_names: Iterable[str], base64_images: Iterable[str]
    ):
        """
        Uploads a base64-encoded image to Azure Blob Storage.

        Args:
            connection_string (str): Connection string to Azure Blob Storage.
            container_name (str): Name of the container.
            blob_name (str): Name of the blob (including extension, e.g., 'image.png').
            base64_image (str): The base64-encoded image string.

        Returns:
            str: URL of the uploaded blob.
        """

        container_client = self.client.get_container_client(self.container_name)

        for blob_name, base64_image in zip(blob_names, base64_images):
            # Decode the base64 image
            image_data = base64.b64decode(base64_image)

            # Create a BlobClient
            blob_client = container_client.get_blob_client(blob_name)

            # Upload the image
            blob_client.upload_blob(
                image_data, overwrite=True, content_type="image/png"
            )

        return
