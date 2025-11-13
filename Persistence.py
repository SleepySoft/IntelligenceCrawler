import os
import re
import copy
import json
import shutil
import hashlib
import logging
import markdown
import requests
import html2text
import traceback
import lxml.etree
import unicodedata
from uuid import uuid4
from bs4 import BeautifulSoup, Tag
from abc import ABC, abstractmethod
from collections import defaultdict
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Optional, Literal

from IntelligenceCrawler.Extractor import ExtractionResult

# try:
#     from weasyprint import HTML
# except Exception as e:
#     HTML = None
#     print(str(e))
#     print('1 - Download https://www.msys2.org/#installation')
#     print('2 - pacman -S mingw-w64-x86_64-pango')
#     print('3 - python3 -m pip install weasyprint')

# Export as PDF feature fail.
HTML = None

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------

IMAGE_PATTERN = re.compile(r'!\[(.*?)\]\((.*?)\)')


def localize_markdown_image(markdown_content: str, base_url: str, image_dir: str):
    """
    Downloads images referenced in markdown and rewrites URLs to local paths.
    (下载 Markdown 中引用的图片，并将 URL 重写为本地路径。)
    """
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
                # 确保在下载前进行依赖检查
                if 'requests' not in globals():
                    raise ImportError("The 'requests' module is required for image downloading.")

                response = requests.get(absolute_url, stream=True, timeout=15)
                response.raise_for_status()  # 检查 HTTP 状态码

                with open(local_path, 'wb') as f:
                    # 优化: 仅在成功下载后才设置 download_success = True
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            logger.info(f"Successfully saved to: {local_path}", 3)

            # 只有当成功保存或文件已存在时，才标记本次操作成功
            download_success = True

            # 4. 重写 Markdown 引用：使用相对于当前工作目录的路径
            # 注意：这里返回的路径是相对于脚本运行目录的相对路径，确保在 Markdown 中能够被正确解析。
            return f"![{alt_text}]({os.path.join(image_dir, filename)})"

        except Exception as e:
            logger.error(f"[Error] Failed to download image from {absolute_url}: {e}", 3)
            # 下载失败，返回原始 URL 引用
            return match.group(0)

    rewritten_markdown = IMAGE_PATTERN.sub(image_replacer, markdown_content)

    # 这里的 download_success 只有在至少一张图片成功下载/被缓存时才是 True
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
    # 1. Do not save if extraction failed or content is empty
    if not result.success:
        logger.error(f"SKIPPING (Failure): {url}. Reason: {result.error}")
        return

    if not result.markdown_content or not result.markdown_content.strip():
        logger.error(f"SKIPPING (No Content): {url}. Markdown is empty.")
        return

    # 初始化可能在 try 块中创建的变量
    base_file_path = None
    image_dir = None

    try:
        # 2. 准备文件路径
        base_file_path = prepare_base_file_path(url, result.metadata, root_dir)

        md_filepath = f"{base_file_path}.md"
        meta_filepath = f"{base_file_path}.meta.json"
        markdown_content = result.markdown_content

        # 3. 图片本地化处理
        if save_image:
            image_dir = f"{base_file_path}.img"
            os.makedirs(image_dir, exist_ok=True)

            rewritten_markdown, download_success = (
                localize_markdown_image(result.markdown_content, url, image_dir))

            if download_success:
                markdown_content = rewritten_markdown
            else:
                # Image parse fail, remove dir.
                logger.info(f"Image localization failed completely for {url}. Removing directory: {image_dir}", 2)
                try:
                    # 确保只在 image_dir 变量被设置且存在时尝试删除
                    if image_dir and os.path.exists(image_dir):
                        # 如果 directory 不为空，shutil.rmtree 是必要的
                        shutil.rmtree(image_dir)
                        logger.info(f"Successfully removed empty/failed image directory: {image_dir}", 2)
                except OSError as e:
                    logger.error(f"Failed to remove image directory {image_dir}: {e}", 2)
                # markdown_content 保持为原始 content，继续保存主文件

        # 4. 保存 Markdown content
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {result.metadata.get('title', 'N/A')}\n\n")
            f.write(f"**Source:** <{url}>\n")

            pub_date = result.metadata.get('date')
            if pub_date:
                f.write(f"**Published:** {pub_date}\n")

            f.write("\n---\n\n")
            f.write(markdown_content)

        # 5. 保存 Metadata
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            # Add the source URL to the metadata for reference
            result.metadata['__source_url'] = url
            # Add the save path for reference
            result.metadata['__saved_md_path'] = md_filepath

            # 使用 default=str 确保像 datetime 对象这样的非 JSON 可序列化对象被正确处理
            json.dump(result.metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"[Handler] SAVED: {url}\n    -> {md_filepath}")

    except Exception as e:
        logger.error(f"[Handler] CRITICAL ERROR saving {url}: {e}")
        # 如果在创建路径后发生错误，尝试清理图片目录（以防创建了一半）
        if base_file_path and save_image and image_dir and os.path.exists(image_dir):
            logger.info(f"Attempting to clean up image directory after critical error: {image_dir}", 2)
            try:
                shutil.rmtree(image_dir)
            except OSError as clean_up_e:
                logger.error(f"Failed cleanup of image directory {image_dir}: {clean_up_e}", 2)

        traceback.print_exc()


# ----------------------------------------------------------------------------------------------------------------------
# --- PDF Generation Function ---

def aggressive_clean_html(html_content: str) -> str:
    """
    使用 BeautifulSoup 积极清理 HTML，移除脚本和样式等可能导致 WeasyPrint 崩溃的元素。
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # 1. 移除脚本和样式，这是最常见的崩溃源
        for script_or_style in soup(["script", "style", "noscript"]):
            script_or_style.decompose()

        # 2. 移除注释
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and '<!--' in text):
            # 使用 .extract() 或其他方法移除注释
            pass

            # 3. 返回清理后的内容（作为字符串）
        return str(soup)
    except Exception as e:
        logger.error(f"HTML aggressive cleaning failed. Error: {e}")
        return html_content  # 如果清理失败，返回原内容


# --- 辅助函数：尝试保存 PDF ---
def save_pdf_attempt(content_string: str, filename: str, base_url: str, metadata: dict, is_fallback=False) -> bool:
    """
    封装 WeasyPrint 调用，隔离渲染异常，并在失败时记录日志。
    """
    try:
        # 重要的改动：使用 os.getcwd() 作为 base_url 的默认值
        html_doc = HTML(
            string=content_string,
            base_url=base_url if base_url else os.getcwd()
        )
        html_doc.write_pdf(filename, metadata=metadata)
        return True
    except Exception as e:
        attempt_name = "Plain Text" if is_fallback else "Cleaned HTML"
        # 仅记录关键错误信息，避免打印巨大的堆栈
        logger.error(f"WeasyPrint rendering failed ({attempt_name}). Reason: {type(e).__name__}: {str(e)[:100]}...")
        return False


# --- 改造后的主函数 ---

def save_extraction_result_as_pdf(result: ExtractionResult, root_dir: str) -> bool:
    """
    Generates a PDF file from raw HTML content, using WeasyPrint with
    a robust fallback mechanism to ensure file generation whenever possible.

    Args:
        :param result: The ExtractionResult object containing content and metadata.
        :param root_dir: The root dir of saving file.

    Returns:
        True if the PDF was successfully created, False otherwise.
    """
    base_file_path = prepare_base_file_path(result.metadata.get('url', str(uuid4())[:8]), result.metadata, root_dir)
    filename = base_file_path + '.pdf'

    # --- 1. 准备元数据结构 ---
    weasyprint_metadata = {}
    weasyprint_metadata['title'] = result.metadata.get('title', os.path.basename(filename))
    weasyprint_metadata['author'] = result.metadata.get('author', 'Document Extractor')
    weasyprint_metadata['subject'] = result.metadata.get('description', 'Extracted Content')

    try:
        metadata_to_dump = {
            k: v for k, v in result.metadata.items()
            if k not in ['title', 'author', 'description']
        }
        metadata_str = json.dumps(metadata_to_dump, default=str, ensure_ascii=False)
        weasyprint_metadata['creator'] = metadata_str
    except Exception as e:
        weasyprint_metadata['creator'] = "Metadata serialization failed."
        logger.error(f"Warning: Could not serialize non-standard metadata. Error: {e}")

    try:
        # 解码原始内容
        raw_content = result.raw_content.decode('utf-8')
        if not raw_content.strip():
            logger.info(f"Warning: Raw HTML content is empty after decoding to save to {filename}")
            return False

        # --- 尝试方案 A: 积极清理后的 HTML (最大限度保留格式) ---
        cleaned_html = aggressive_clean_html(raw_content)

        if save_pdf_attempt(cleaned_html, filename, result.metadata.get('url'), weasyprint_metadata, is_fallback=False):
            logger.info(f"Successfully generated PDF (Plan A - Cleaned HTML): {filename}")
            return True

        # --- 如果方案 A 失败，进入方案 B: 纯文本回退 (最大限度保障生成) ---
        logger.warning(f"Plan A failed. Attempting Plan B (Plain Text Fallback) for {filename}. Format loss expected.")

        # 1. 提取所有可见文本
        soup = BeautifulSoup(raw_content, 'html.parser')
        plain_text = soup.get_text('\n', strip=True)  # 使用换行符作为分隔符

        # 2. 将纯文本包装到最小的 HTML 结构中，使用 <pre> 标签保留换行和空格
        fallback_html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{weasyprint_metadata['title']}</title></head>
<body>
    <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: sans-serif;">{plain_text}</pre>
</body>
</html>
"""
        # 3. 尝试保存纯文本 PDF
        if save_pdf_attempt(fallback_html, filename, result.metadata.get('url'), weasyprint_metadata, is_fallback=True):
            logger.info(f"Successfully generated PDF (Plan B - Plain Text Fallback): {filename}")
            return True

        # --- 方案 A 和 B 都失败 ---
        logger.error(f"FATAL: Both Plan A and Plan B failed for {filename}. PDF could not be generated.")
        return False

    except Exception as e:
        # 捕获外部异常（如文件系统错误、解码错误等）
        logger.error(f"An unexpected external error occurred during PDF generation for {filename}: {e}")
        print(traceback.format_exc())
        return False
