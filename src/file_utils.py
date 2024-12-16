import base64
import hashlib
import io
import os
from typing import List

import fitz  # PyMuPDF


def pdf_blob_to_pymupdf_doc(blob: bytes) -> fitz.Document:
    """
    Converts a PDF byte blob into a PyMuPDF Document object.

    Args:
        blob (bytes): A byte blob representing a PDF file.

    Returns:
        fitz.Document: The PyMuPDF Document object created from the byte blob.
    """
    return fitz.open(filetype="pdf", stream=blob)


def extract_single_image(doc: fitz.Document, xref: int) -> fitz.Pixmap:
    """
    Extracts a single image from the document given its xref.
    Converts it to RGB format if necessary and returns the Pixmap.

    Args:
        doc (fitz.Document): The PyMuPDF document object containing the image.
        xref (int): The xref number of the image to extract.

    Returns:
        fitz.Pixmap: A Pixmap object of the extracted image.
    """
    pix = fitz.Pixmap(doc, xref)  # Create the Pixmap for the image
    if pix.n > 3:  # Convert to RGB if necessary
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix


def page_extract_images(page: fitz.Page) -> List[fitz.Pixmap]:
    """
    Extracts all images on a given page as Pixmap objects.

    Args:
        page (fitz.Page): A single page of a PyMuPDF document.

    Returns:
        List[fitz.Pixmap]: A list of Pixmap objects for each image on the page.
    """
    images = []
    doc: fitz.Document = page.parent

    for _, img in enumerate(page.get_images()):
        xref = img[0]
        pix = extract_single_image(doc, xref)
        images.append(pix)
    return images


def get_images_as_base64(page: fitz.Page) -> List[str]:
    """
    Converts all images on a given page to base64-encoded strings.

    Args:
        page (fitz.Page): A single page of a PyMuPDF document.

    Returns:
        List[str]: A list of base64-encoded strings, each representing an image on the page.
    """
    images_base64 = []
    images = page_extract_images(page)  # Get all images on the page

    for pix in images:
        # Convert Pixmap to PNG format in-memory
        img_buffer = io.BytesIO(pix.tobytes("png"))
        # Encode PNG binary data as base64 string
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        images_base64.append(img_base64)

    return images_base64


def create_file_metadata_from_path(file_path):
    """
    Create metadata for a document file.

    Parameters:
    - file_path (str): The file path to the PDF document.

    Returns:
    - dict: Metadata dictionary containing the document title, file name, and SHA-256 hash.
    """
    # Extract the file name without the directory path and extension
    title = os.path.splitext(os.path.basename(file_path))[0]
    file_name = os.path.basename(file_path)

    # Calculate SHA-256 hash to uniquely identify the file
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid memory overload with large files
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    # Generate the hash in hexadecimal format
    file_hash = sha256_hash.hexdigest()

    return {"title": title, "file": file_name, "file_hash": file_hash}


def create_file_metadata_from_bytes(file_bytes: bytes, file_name: str, title=None):
    """
    Create metadata for a document file using the file contents in bytes.

    Parameters:
    - file_bytes (bytes): The bytes content of the document file.
    - file_name (str): The file name of the document.
    - title (str, optional): The title of the document. If not provided, it will be inferred from the file_name.

    Returns:
    - dict: Metadata dictionary containing the document title, file name, and SHA-256 hash.
    """
    # If title is not provided, infer it from the file_name
    if title is None:
        title = os.path.splitext(file_name)[0]

    # Calculate SHA-256 hash to uniquely identify the file
    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_bytes)
    file_hash = sha256_hash.hexdigest()

    return {"title": title, "file": file_name, "file_hash": file_hash}
