"""
Implement duplicating rules.
If a file is a duplicate, skip processing it
"""

import hashlib
import json
from typing import Any, Dict, List

from azure.storage.blob import BlobServiceClient
from loguru import logger

from src.azure_container_client import BaseAzureContainerClient


class DuplicateChecker(BaseAzureContainerClient):
    """
    Simple duplicate checks using key pair storage in JSON format
    """

    def __init__(self, client: BlobServiceClient, container_name: str):
        super().__init__(client, container_name)
        self.blob_name = "known_files.json"

        # Initialize default structure
        self.known_dict: Dict[str, List[str]] = {
            "known_titles": [],
            "known_hashes": [],
            "known_file_names": [],
        }

        # Try to load existing knowledge
        try:
            if known_files_bytes := self.download_file(self.blob_name):
                # Decode bytes to string and parse JSON
                known_files_str = known_files_bytes.decode("utf-8")
                loaded_knowledge = json.loads(known_files_str)

                # Update with loaded data, maintaining default structure
                self.known_dict.update(loaded_knowledge)

                logger.info(f"Successfully loaded blob {self.blob_name}")
            else:
                logger.info(f"No existing blob {self.blob_name}")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.blob_name}: {e}")
        except UnicodeDecodeError as e:
            logger.error(f"Error decoding bytes from {self.blob_name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading knowledge file: {e}")

    def save(self):
        """
        Save current knowledge to blob storage

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Update knowledge dictionary with current lists
            self.known_dict.update(
                {
                    "known_titles": self.known_dict["known_titles"],
                    "known_hashes": self.known_dict["known_hashes"],
                    "known_file_names": self.known_dict["known_file_names"],
                }
            )

            # Convert to JSON string and encode to bytes
            json_data = json.dumps(self.known_dict, indent=2)
            json_bytes = json_data.encode("utf-8")

            # Upload to blob storage

            container_client = self.client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(self.blob_name)
            try:
                blob_client.upload_blob(
                    json_bytes, overwrite=True, content_type="application/json"
                )

                logger.info(f"Successfully saved knowledge to {self.blob_name}")
            except Exception as e:
                logger.error(f"Error uploading known file data to {self.blob_name}")
            finally:
                container_client.close()
                blob_client.close()

        except Exception as e:
            logger.error(f"Error saving knowledge file: {e}")

    def update(self, file_hash=None, file_name=None, title=None):
        """
        Add new file_hash or file_name or title to the known files json
        """

        def add_to_known_dict(key, value):
            if value:
                self.known_dict[key] = list(set(self.known_dict[key] + [value]))

        add_to_known_dict("known_hashes", file_hash)
        add_to_known_dict("known_file_names", file_name)
        add_to_known_dict("known_titles", title)

    def duplicate_by_title(self, title: str, case_sensitive=False):
        if case_sensitive:
            return title in self.known_dict["known_titles"]
        return title.lower() in [
            known_title.lower() for known_title in self.known_dict["known_titles"]
        ]

    def duplicate_by_hash(self, file_hash: str):
        return file_hash in self.known_dict["known_hashes"]

    def duplicate_by_file_name(self, file_name: str):
        return file_name in self.known_dict["known_file_names"]

    @staticmethod
    def create_hash(byte_data: bytes) -> str:
        return hashlib.sha256(byte_data).hexdigest()
