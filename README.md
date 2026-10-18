# 🌐 PageSift AI

> **A high-throughput Map-Reduce LLM summarization pipeline that processes long-form web content concurrently with intelligent caching and structured synthesis.**

---

## 📌 Architecture Overview

When handling large webpages (Wikipedia articles, financial prospectuses, long blogs), passing the raw HTML directly to an LLM introduces three major bottlenecks:
1. **Context Window Saturation & Cost**: Excessive token usage and context bloat.
2. **"Lost in the Middle" Effect**: LLMs often miss nuances buried deep inside 20,000-word documents.
3. **High Latency**: Sequential processing is slow.

This project implements a classic **Map-Reduce** architectural pattern:

```
[Target URL]
     │
     ▼
[Web Scraper & Cleaner] ─── (Strips nav, script, ads; extracts main article body)
     │
     ▼
[Sentence-Aware Chunker] ─── (Splits text preserving sentence boundaries, ~1,500 chars)
     │
     ├────────────────────────────────────────┐
     ▼                                        ▼
[Chunk 1 (Map Worker)]             [Chunk N (Map Worker)]
 (Extracts facts & takeaways)       (Extracts facts & takeaways)
     │                                        │
     └───────────────────┬────────────────────┘
                         ▼
             [Hierarchical Synthesizer (Reduce)]
                         │
                         ▼
             [Structured Executive Brief]
                         │
                         ▼
              [Disk-Backed MD5 Cache]
```

---

## ✨ Key Features

- **Map-Reduce Parallelism**: Uses Python's `concurrent.futures.ThreadPoolExecutor` to process document chunks simultaneously, cutting response latency by up to 60%.
- **Intelligent Fast-Path**: Automatically skips Map-Reduce for short pages (< 2,000 characters), summarizing directly in a single pass to save API calls.
- **Noise Reduction**: Pre-processes HTML with `BeautifulSoup` to strip navigation, ads, headers, footers, and scripts, keeping only semantic content.
- **Sentence-Bounded Chunking**: Unlike naive character splitters, `smart_chunk()` splits cleanly at punctuation boundaries (`.!?`) so context is never cut mid-sentence.
- **MD5 Disk Caching**: Caches final summaries by URL hash in `cache/` to ensure zero redundant API queries on repeated runs.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **LLM Engine**: Groq API (`llama-3.3-70b-versatile`) via OpenAI SDK
- **Web Scraping**: `requests`, `BeautifulSoup4`
- **Concurrency**: `concurrent.futures.ThreadPoolExecutor`
- **Environment**: `python-dotenv`

---

## 🚀 Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/revanth-exe/page-sift-ai.git
cd page-sift-ai
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```
Add your Groq API key:
```env
GROQ_API_KEY=gsk_...
```

### 3. Run
```bash
python app.py
```

---

## 🎯 Interview Cheat-Sheet: How to Explain This Project

### 1. The Elevator Pitch
> *"I built PageSift AI, a scalable Map-Reduce summarization engine using Groq and Llama 3.3. When summarizing long web articles, sending the entire text to an LLM causes the 'Lost in the Middle' problem and spikes token latency. My pipeline cleans the HTML, chunks the text along sentence boundaries, summarizes each chunk in parallel using a thread pool, and then performs a hierarchical reduce step to generate a unified executive report with persistent MD5 caching."*

### 2. Common Interview Questions & How to Answer

#### Q: Why Map-Reduce instead of simply passing the whole text into a 128k context window?
- **Answer**: *"While modern models have 128k+ context windows, processing huge prompts is computationally expensive, slow, and prone to attention degradation ('Lost in the Middle'). Map-Reduce allows parallel execution across workers, guarantees that every section of the document receives focused attention, and significantly cuts time-to-first-token."*

#### Q: How do you handle rate limits when running chunks in parallel?
- **Answer**: *"I capped `max_workers=3` to balance concurrency speed against the token-per-minute (TPM) limits on Groq's free and standard tiers. In production, I would add exponential backoff retries using `tenacity` or a distributed task queue like Celery with Redis."*
