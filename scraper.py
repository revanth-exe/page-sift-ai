"""
scraper.py - Web Content Extraction Module
Extracts clean, readable text from arbitrary web pages while filtering
out boilerplate elements like navigation, advertisements, footers, and scripts.
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional


def fetch_website_contents(url: str, timeout: int = 15) -> str:
    """
    Fetches the web page at `url`, strips non-content tags, and returns
    cleaned, normalized text.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Strip noisy and non-semantic elements
    unwanted_tags = [
        "script", "style", "noscript", "nav", "footer", "header",
        "aside", "form", "svg", "button", "iframe", "figure"
    ]
    for tag in soup(unwanted_tags):
        tag.decompose()

    # Prioritize primary content containers if available
    # 1. Wikipedia main body
    if "wikipedia.org" in url:
        content_container = soup.find("div", {"id": "mw-content-text"})
    # 2. HTML5 <article> tag
    elif soup.find("article"):
        content_container = soup.find("article")
    # 3. HTML5 <main> tag
    elif soup.find("main"):
        content_container = soup.find("main")
    # 4. Fallback to body
    else:
        content_container = soup.body or soup

    if not content_container:
        return ""

    # Extract text with space separator and strip leading/trailing whitespace
    text = content_container.get_text(separator=" ", strip=True)
    return text
