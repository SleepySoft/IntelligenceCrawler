
import hashlib
import logging
import os
import re
import copy
import json
from uuid import uuid4

import requests
import html2text
import traceback
import lxml.etree
import unicodedata
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Tag
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Any, Optional, Literal
# External dependencies for PDF generation and Markdown processing
# These need to be installed: pip install reportlab markdown pydantic
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
import markdown

from IntelligenceCrawler.Extractor import ExtractionResult

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------

IMAGE_PATTERN = re.compile(r'!\[(.*?)\]\((.*?)\)')

def localize_markdown_image(markdown_content: str, base_url: str, image_dir: str):
    """
    Downloads images referenced in markdown and rewrites URLs to local paths.
    (下载 Markdown 中引用的图片，并将 URL 重写为本地路径。)
    """
    if not requests:
        logger.error("[Error] 'requests' not installed. Cannot download images.", 2)
        return markdown_content, False

    try:
        # 确保图片保存目录存在
        os.makedirs(image_dir, exist_ok=True)
        logger.info(f"Image directory created/checked: {image_dir}", 2)
    except OSError as e:
        logger.error(f"[Error] Could not create image directory {image_dir}: {e}", 2)
        return markdown_content, False

    download_success = False

    def image_replacer(match):
        nonlocal download_success
        alt_text = match.group(1)
        original_url = match.group(2)

        # 1. 解析绝对 URL
        absolute_url = urljoin(base_url, original_url)

        # 2. 生成唯一文件名
        url_path = urlparse(absolute_url).path
        extension = os.path.splitext(url_path)[1].lower()
        if not extension or len(extension) > 5 or '.' not in extension:
            # 尝试从响应头或 URL 参数中获取更准确的 MIME 类型，这里简化为默认 .jpg
            extension = '.jpg'

            # 使用 URL 的 SHA256 哈希值作为唯一文件名
        url_hash = hashlib.sha256(absolute_url.encode('utf-8')).hexdigest()[:10]
        filename = f"{url_hash}{extension}"
        local_path = os.path.join(image_dir, filename)

        # 3. 下载图片
        try:
            logger.info(f"Downloading: {absolute_url}", 3)

            # 如果文件已存在，则跳过下载 (简单的缓存机制)
            if os.path.exists(local_path):
                logger.info(f"File already exists: {local_path}. Skipping download.", 3)
            else:
                response = requests.get(absolute_url, stream=True, timeout=15)
                response.raise_for_status()  # 检查 HTTP 状态码

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            logger.info(f"Successfully saved to: {local_path}", 3)
            download_success = True

            # 4. 重写 Markdown 引用：使用相对于当前工作目录的路径
            return f"![{alt_text}]({os.path.join(image_dir, filename)})"

        except Exception as e:
            logger.error(f"[Error] Failed to download image from {absolute_url}: {e}", 3)
            # 下载失败，返回原始 URL 引用 (WeasyPrint 会尝试再次下载)
            return match.group(0)

            # 使用正则表达式替换函数处理所有图片链接

    rewritten_markdown = IMAGE_PATTERN.sub(image_replacer, markdown_content)

    return rewritten_markdown, download_success


# ----------------------------------------------------------------------------------------------------------------------

def _slugify_filename(text: str) -> str:
    """
    Sanitize a string to be used as a valid filename.
    (e.g., "My Post / 1? (我的帖子)" -> "my-post-1-我的帖子")

    :param text: The string to sanitize.
    :return: A filesystem-safe string.
    """
    if not text:
        return "untitled"

    # 1. 转换为小写
    text = str(text).lower()

    # 2. 移除特殊字符 (保留 Unicode 字母数字、空格、横线)
    # 删除了 flags=re.ASCII，\w 现在会匹配中文字符
    text = re.sub(r'[^\w\s-]', '', text)

    # 3. 将连续的空格、下划线、横线替换为单个横线
    text = re.sub(r'[\s_-]+', '-', text).strip('-')

    # 4. 截断到合理长度
    text = text[:100]

    # 5. 再次清理，防止截断后留下末尾的横线
    text = text.strip('-')

    # 6. 如果清理后字符串为空，返回一个默认值
    if not text:
        return "untitled"

    return text


def _get_safe_basename(url: str, metadata: Dict[str, Any]) -> str:
    """
    Create a safe base filename (no extension) for the article.
    It tries to find a human-readable name and falls back to a hash.

    :param url: The source URL of the article.
    :param metadata: The extracted metadata dictionary.
    :return: A filesystem-safe base filename.
    """

    # Strategy 1: Use the title if it exists
    title = metadata.get('title')
    if title:
        slug = _slugify_filename(title)
        # Avoid empty slugs if the title was just symbols
        if slug:
            return slug

    # Strategy 2: Use the last part of the URL path
    try:
        path_filename = os.path.basename(urlparse(url).path)
        if path_filename:
            # Remove extension (e.g., .html, .aspx)
            base = os.path.splitext(path_filename)[0]
            slug = _slugify_filename(base)
            if slug:
                return slug
    except Exception:
        pass  # Fallback to hash

    # Strategy 3: (Fallback) Use a hash of the URL for a guaranteed unique name
    return hashlib.md5(url.encode('utf-8')).hexdigest()


def prepare_base_file_path(url: str, metadata: Dict[str, Any], root_dir: str):
    # Determine Category (Folder) from domain
    parsed_url = urlparse(url)
    domain = parsed_url.netloc or "unknown_domain"
    # Sanitize domain to remove ports (e.g., "localhost:8000")
    domain = domain.split(':', 1)[0]

    # 3. Determine safe filename
    base_filename = _get_safe_basename(url, metadata)

    # Define full file paths
    # Structure: CRAWLER_OUTPUT / <domain> / <base_filename>
    article_dir = os.path.join(root_dir, domain)

    # Create the directory if it doesn't exist
    os.makedirs(article_dir, exist_ok=True)

    base_file_path = os.path.join(article_dir, f"{base_filename}")
    return base_file_path

def save_extraction_result_as_md(
        url: str,
        result: ExtractionResult,
        save_image: bool = False,
        root_dir: str = 'CRAWLER_OUTPUT'):
    """
    This is the content_handler implementation for the CrawlPipeline.

    It saves the extracted article markdown and metadata to a
    categorized folder structure:

    CRAWLER_OUTPUT/
        └── <domain.com>/
            ├── <article-slug>.md
            └── <article-slug>.meta.json

    :param url: The source URL of the article.
    :param result: The ExtractionResult object from the extractor.
    :param save_image: True to download and save image else False.
    :param root_dir: The root dir to save articles.
    """
    # Do not save if extraction failed or content is empty
    if not result.success:
        logger.error(f"SKIPPING (Failure): {url}. Reason: {result.error}")
        return

    if not result.markdown_content or not result.markdown_content.strip():
        logger.error(f"SKIPPING (No Content): {url}. Markdown is empty.")
        return

    try:
        base_file_path = prepare_base_file_path(url, result.metadata, root_dir)

        md_filepath = f"{base_file_path}.md"
        meta_filepath = f"{base_file_path}.meta.json"
        markdown_content = result.markdown_content

        if save_image:
            os.makedirs(image_dir := f"{base_file_path}", exist_ok=True)
            rewritten_markdown, download_success = (
                localize_markdown_image(result.markdown_content, url, image_dir))
            if download_success:
                markdown_content = rewritten_markdown

        # Save Markdown content
        # We'll add a simple header to the Markdown file
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {result.metadata.get('title', 'N/A')}\n\n")
            f.write(f"**Source:** <{url}>\n")

            pub_date = result.metadata.get('date')
            if pub_date:
                f.write(f"**Published:** {pub_date}\n")

            f.write("\n---\n\n")
            f.write(markdown_content)

        # 7. Save Metadata
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            # Add the source URL to the metadata for reference
            result.metadata['__source_url'] = url
            # Add the save path for reference
            result.metadata['__saved_md_path'] = md_filepath

            json.dump(result.metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"[Handler] SAVED: {url}\n    -> {md_filepath}")

    except Exception as e:
        logger.error(f"[Handler] CRITICAL ERROR saving {url}: {e}")
        traceback.print_exc()


# ----------------------------------------------------------------------------------------------------------------------
# --- PDF Generation Function ---

def save_extraction_result_as_pdf(result: ExtractionResult, root_dir: str) -> bool:
    """
    Generates a PDF file from an ExtractionResult object, adding metadata
    to the document properties instead of the body.

    Args:
        :param result: The ExtractionResult object containing content and metadata.
        :param root_dir: The root dir of saving file.

    Returns:
        True if the PDF was successfully created, False otherwise.
    """
    base_file_path = prepare_base_file_path(result.metadata.get('url', str(uuid4())[:8]), result.metadata, root_dir)
    filename = base_file_path + '.pdf'

    try:
        # Check if there is content to print
        if not result.markdown_content:
            print(f"Warning: No content available in ExtractionResult to save to {filename}")
            return False

        # --- 1. Setup Document and Metadata ---
        # Initialize the PDF document template
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Apply metadata to the PDF document properties (not in the body)
        doc.title = result.metadata.get('title', os.path.basename(filename))
        doc.author = result.metadata.get('author', 'Document Extractor')
        doc.subject = result.metadata.get('description', 'Extracted Content')

        # Use the 'creator' field to store the rest of the metadata as a JSON string
        try:
            # Ensure complex types are serialized
            metadata_str = json.dumps(result.metadata, default=str, ensure_ascii=False)
            doc.creator = metadata_str
        except Exception as e:
            # Fallback if metadata serialization fails
            doc.creator = "Metadata serialization failed."
            print(f"Warning: Could not serialize all metadata. Error: {e}")

        Story = []
        styles = getSampleStyleSheet()

        # Define a base style for the body text
        styles.add(ParagraphStyle(
            name='NormalBody',
            parent=styles['Normal'],
            fontName='Helvetica',  # Use a standard font for broad compatibility
            fontSize=12,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceBefore=6,
            spaceAfter=6,
        ))

        # --- 2. Content Conversion and Formatting ---
        content_type = result.metadata.get('content_type', 'markdown').lower()
        content_to_render = result.markdown_content

        if content_type == 'markdown':
            # Convert markdown to basic HTML for ReportLab's Paragraph flowables
            # ReportLab's Paragraph supports a subset of HTML tags (like <b>, <i>, <h1>-<h6>, <br/>)
            html_content = markdown.markdown(content_to_render)
            Story.append(Paragraph(html_content, styles['NormalBody']))

        elif content_type == 'raw_html':
            # Render raw HTML directly
            Story.append(Paragraph(content_to_render, styles['NormalBody']))

        else:  # Default or unknown type: treat as plain text
            # Replace newlines with <br/> tags to ensure line breaks are visible in the PDF
            plain_text_html = content_to_render.replace('\n', '<br/>')
            Story.append(Paragraph(plain_text_html, styles['NormalBody']))

        # Add a final spacer for good measure
        Story.append(Spacer(1, 12))

        # --- 3. Build the PDF ---
        doc.build(Story)
        print(f"Successfully generated PDF: {filename}")
        return True

    except Exception as e:
        # Catch any exception and return False, as required
        print(f"An error occurred during PDF generation for {filename}: {e}")
        return False
