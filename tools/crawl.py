import json
import logging
from typing import Annotated, Optional
from urllib.parse import urlparse

from langchain_core.tools import tool

from app.tools.decorators import log_io

logger = logging.getLogger(__name__)


def is_pdf_url(url: Optional[str]) -> bool:
    if not url:
        return False
    parsed_url = urlparse(url)
    return parsed_url.path.lower().endswith(".pdf")


@tool
@log_io
def crawl_tool(url: Annotated[str, "The url to crawl."]) -> str:
    """
    Placeholder crawler aligned with DeerFlow signature.
    In real use, plug in a proper crawler and return markdown content.
    """
    if is_pdf_url(url):
        logger.info("PDF URL detected, skipping crawling: %s", url)
        return json.dumps(
            {
                "url": url,
                "error": "PDF files cannot be crawled directly. Please download manually.",
                "crawled_content": None,
                "is_pdf": True,
            }
        )
    # Stub content
    return json.dumps({"url": url, "crawled_content": f"[stubbed content for {url}]"})
