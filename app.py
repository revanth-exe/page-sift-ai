"""
app.py - Map-Reduce Web Summarizer AI
Implements a scalable Map-Reduce LLM summarization pipeline with concurrent
chunk processing, disk-backed caching, and structured synthesis.
"""

import os
import re
import time
import hashlib
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI
from scraper import fetch_website_contents

load_dotenv()

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    # Gracefully warn if not yet set
    print("Warning: GROQ_API_KEY is not set in environment or .env file.")

client = OpenAI(
    base_url=BASE_URL,
    api_key=GROQ_API_KEY or "dummy_key_set_in_env"
)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# PROMPTS
# -----------------------------
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are an expert AI research assistant specializing in technical and analytical summarization. "
        "Extract only high-signal insights, empirical facts, and key takeaways. "
        "Ignore noise such as boilerplate, navigation, ads, and repetitions. "
        "Format output cleanly in Markdown."
    )
}

# -----------------------------
# CACHING UTILITIES
# -----------------------------
def hash_url(url: str) -> str:
    """Generate MD5 hash of URL for cache keys."""
    return hashlib.md5(url.strip().lower().encode("utf-8")).hexdigest()


def load_cache(url: str) -> Optional[str]:
    """Load cached summary if it exists."""
    cache_path = os.path.join(CACHE_DIR, f"{hash_url(url)}.txt")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def save_cache(url: str, data: str) -> None:
    """Persist generated summary to disk cache."""
    cache_path = os.path.join(CACHE_DIR, f"{hash_url(url)}.txt")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(data)

# -----------------------------
# TEXT PROCESSING & CHUNKING
# -----------------------------
def clean_text(text: str) -> str:
    """Normalize whitespace and strip unwanted characters."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def smart_chunk(text: str, max_chars: int = 1500) -> List[str]:
    """
    Splits long text into coherent sentence-bounded chunks.
    Ensures that sentence boundaries are preserved and chunks do not split thoughts.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# -----------------------------
# LLM INFERENCE
# -----------------------------
def call_model(messages: list, temperature: float = 0.2) -> str:
    """Wrapper around chat completions with temperature control."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content or ""

# -----------------------------
# MAP-REDUCE PIPELINE
# -----------------------------
def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int) -> str:
    """MAP STEP: Summarizes an individual text chunk."""
    if not chunk.strip():
        return ""

    messages = [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                f"You are processing part {chunk_index + 1} of {total_chunks} of a larger document.\n"
                f"Extract the primary facts, figures, and concepts from the text below.\n\n"
                f"--- CHUNK CONTENT ---\n{chunk}\n\n"
                "Return a bulleted list of 3-5 core takeaways."
            )
        }
    ]
    return call_model(messages)


def summarize_parallel(chunks: List[str], max_workers: int = 3) -> List[str]:
    """Executes chunk summarization concurrently using a thread pool."""
    total = len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(summarize_chunk, chunk, idx, total)
            for idx, chunk in enumerate(chunks)
        ]
        return [f.result() for f in futures]


def combine_summaries(summaries: List[str], original_url: str) -> str:
    """REDUCE STEP: Synthesizes intermediate chunk summaries into an executive report."""
    valid_summaries = [s.strip() for s in summaries if s and len(s.strip()) > 15]

    if not valid_summaries:
        return "Failed to synthesize summaries: No valid chunk summaries returned."

    combined_input = "\n\n".join(
        f"[Section {i+1}]:\n{s}" for i, s in enumerate(valid_summaries)
    )

    messages = [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                f"The following are intermediate section summaries extracted from: {original_url}\n\n"
                f"{combined_input}\n\n"
                "Synthesize this into a cohesive, high-quality Executive Summary with this structure:\n"
                "1. **Core Overview** (2-3 sentences explaining what this content is about)\n"
                "2. **Key Takeaways & Findings** (Bullet points with bold lead-ins)\n"
                "3. **Technical / Domain Highlights** (Specific data, tools, or conclusions)\n"
                "4. **Bottom Line Conclusion** (One sentence final takeaway)\n"
            )
        }
    ]
    return call_model(messages, temperature=0.3)


def summarize_website(url: str, force_refresh: bool = False) -> str:
    """
    End-to-end pipeline:
    1. Check cache
    2. Scrape & clean
    3. Determine strategy (single-pass vs map-reduce)
    4. Execute & cache
    """
    start_time = time.time()

    # Step 1: Cache Check
    if not force_refresh:
        cached = load_cache(url)
        if cached:
            print(f"⚡ [Cache Hit] Loaded cached summary for {url}")
            return cached

    # Step 2: Web Scraping
    print(f"🌐 Fetching content from: {url}")
    try:
        raw_text = fetch_website_contents(url)
    except Exception as e:
        return f"❌ Error fetching website: {e}"

    if not raw_text or len(raw_text.strip()) < 50:
        return "❌ Warning: No meaningful content could be extracted from this URL."

    # Clean and cap at 30,000 characters to prevent excessive tokens
    cleaned = clean_text(raw_text)[:30000]
    print(f"🧹 Cleaned text length: {len(cleaned)} characters")

    # Step 3: Fast-path for short articles
    if len(cleaned) <= 2000:
        print("⚡ Short document detected: Using single-pass direct summarization...")
        messages = [
            SYSTEM_PROMPT,
            {
                "role": "user",
                "content": f"Summarize the following web content from {url}:\n\n{cleaned}"
            }
        ]
        final_summary = call_model(messages)
    else:
        # Step 4: Chunking
        chunks = smart_chunk(cleaned, max_chars=1800)
        print(f"✂️  Segmented into {len(chunks)} sentence-bounded chunks")

        # Step 5: Map Phase (Parallel)
        print(f"🚀 Running Map Phase: Summarizing {len(chunks)} chunks concurrently...")
        chunk_summaries = summarize_parallel(chunks, max_workers=3)

        # Step 6: Reduce Phase (Combine)
        print("🧠 Running Reduce Phase: Synthesizing final structured summary...")
        final_summary = combine_summaries(chunk_summaries, url)

    # Step 7: Save to disk cache
    save_cache(url, final_summary)
    elapsed = time.time() - start_time
    print(f"✅ Completed in {elapsed:.2f} seconds.")

    return final_summary


# -----------------------------
# CLI INTERFACE
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  🌐 Web Summarizer AI - Map-Reduce Pipeline")
    print("=" * 60)
    target_url = input("Enter website URL to summarize: ").strip()

    if not target_url:
        target_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        print(f"Defaulting to: {target_url}")

    result = summarize_website(target_url)
    print("\n" + "=" * 60)
    print("  📋 FINAL EXECUTIVE SUMMARY")
    print("=" * 60 + "\n")
    print(result)
