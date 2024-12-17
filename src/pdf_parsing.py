"""
Define a pdf parser object that extract texts and images from doc
while maintaining page information
"""

import base64
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fitz import Document, Matrix, Page
from loguru import logger
from pydantic import BaseModel

from src.file_utils import get_images_as_base64, page_extract_images


class FileText(BaseModel):
    """
    Represents a page of text
    """

    page_no: int
    text: Optional[str]


class FileImage(BaseModel):
    """
    Represent an image
    """

    page_no: int
    image_no: int
    image_base64: str


@dataclass
class PageStats:
    """
    Document statistics on the number of pages grouped by
        whether they contain or not contain any texts or images
    """

    text_yes_image_yes: int = 0
    text_yes_image_no: int = 0
    text_no_image_yes: int = 0
    text_no_image_no: int = 0

    def update(self, has_text: bool, has_images: bool) -> None:
        if has_text and has_images:
            self.text_yes_image_yes += 1
        elif has_text:
            self.text_yes_image_no += 1
        elif has_images:
            self.text_no_image_yes += 1
        else:
            self.text_no_image_no += 1

    def log_summary(self, doc_metadata: dict) -> None:
        logger.info(f"File metadata: {doc_metadata}")
        logger.info(
            "\n"
            "|                     | Has Images         | No Images          |\n"
            "|---------------------|--------------------|--------------------|\n"
            f"| **Has Text**       | {self.text_yes_image_yes:>18} | {self.text_yes_image_no:>18} |\n"
            f"| **No Text**        | {self.text_no_image_yes:>18} | {self.text_no_image_no:>18} |"
        )


def doc_is_ppt(doc: Document) -> bool:
    """Return True if pdf document is a PowerPoint export"""
    return any(
        "PowerPoint" in doc.metadata.get(field, "") for field in ["creator", "producer"]
    )


def page_to_base64(page: Page, format: str = "png", scale: int = 2) -> str:
    """Convert whole page to base64 image"""
    return base64.b64encode(
        page.get_pixmap(matrix=Matrix(scale, scale)).tobytes(format)
    ).decode()


def is_drawing_not_visible(item: dict) -> bool:
    """Determine if an item is not visible based on its attributes"""
    no_fill = item.get("fill") is None or item.get("fill_opacity", 1.0) == 0.0
    no_stroke = (
        item.get("color") is None
        or item.get("stroke_opacity", 1.0) == 0.0
        or item.get("width", 1.0) <= 0
    )
    return no_fill and no_stroke


def get_page_drawings_stats(
    page: Page, get_invisible_elements: bool = True
) -> Dict[str, int]:
    """Count drawings by type: curve, line, quad, rectangle"""
    stats = dict()
    for drawing in page.get_drawings():
        if not get_invisible_elements and is_drawing_not_visible(drawing):
            continue
        for item in drawing["items"]:
            stats[item[0]] = stats.get(item[0], 0) + 1
    return stats


def is_infographic_page(page: Page) -> bool:
    """Check if page contains multiple visual components"""
    stats = get_page_drawings_stats(page, get_invisible_elements=False)

    n_elements = sum(v for k, v in stats.items() if k != "re")

    # If the number of lines + curves + quads exceeds this threshold, we flag the whole page as an image
    return n_elements >= 9


def process_page_as_an_image(
    page: Page, page_no: int, stats: PageStats
) -> Tuple[List[FileText], List[FileImage]]:
    """Process a page like the whole page is an image"""
    page_image = FileImage(
        page_no=page_no, image_no=page_no, image_base64=page_to_base64(page, scale=1)
    )
    stats.update(has_text=False, has_images=True)
    return [], [page_image]


def process_regular_page(
    page: Page, page_no: int, stats: PageStats
) -> Tuple[List[FileText], List[FileImage]]:
    """Process regular PDF page with text and images"""
    if is_infographic_page(page):
        logger.info(
            f"Page {page_no} contains multiple visual elements and will be treated as an image"
        )
        return process_page_as_an_image(page, page_no, stats)

    text = page.get_text()
    images_base64 = get_images_as_base64(page)

    # Filter multicolor images
    images_pixmap = page_extract_images(page)
    images_base64 = [
        img
        for img, pixmap in zip(images_base64, images_pixmap)
        if not pixmap.is_unicolor
    ]

    if not text:
        if images_base64 or page.get_drawings():
            logger.info(
                f"Page {page_no} contains no text elements and will be treated as an image"
            )
            return process_page_as_an_image(page, page_no, stats)
        else:
            return [], []

    # Process text and images
    texts = [FileText(page_no=page_no, text=text)]
    images = [
        FileImage(page_no=page_no, image_base64=img, image_no=i)
        for i, img in enumerate(images_base64)
    ]

    stats.update(has_text=bool(text), has_images=bool(images))
    return texts, images


def extract_texts_and_images(
    doc: Document, report: bool = False
) -> Tuple[List[FileText], List[FileImage]]:
    """Extract texts and images from PDF document"""
    stats = PageStats()
    all_texts: List[FileText] = []
    all_images: List[FileImage] = []

    process_fn = process_page_as_an_image if doc_is_ppt(doc) else process_regular_page

    for page_no, page in enumerate(doc):
        texts, images = process_fn(page, page_no, stats)
        all_texts.extend(texts)
        all_images.extend(images)

    if report:
        stats.log_summary(doc.metadata)

    return all_texts, all_images
