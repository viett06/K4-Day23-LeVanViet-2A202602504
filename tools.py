"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random
import re
import threading
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)
from urllib.parse import quote, urlencode

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_LAST_ARXIV_CALL = 0.0
_ARXIV_LOCK = threading.Lock()


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    attempts = max(1, attempts)
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(max(0.0, float(exc.retry_after)), cap)
            else:
                delay = min(base * (2 ** attempt) + random.uniform(0, base), cap)
            time.sleep(delay)


def _clean(text, limit=None):
    text = " ".join(str(text or "").split())
    if limit and len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _clamp(value, low, high, default):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _retryable_get(url, *, params=None, timeout=30, attempts=5, cap=30, before_request=None):
    def call():
        if before_request is not None:
            before_request()
        try:
            response = httpx.get(url, params=params, timeout=timeout)
        except httpx.TransportError as exc:
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code in RETRYABLE_STATUS:
            retry_after = response.headers.get("Retry-After")
            raise RetryableError(f"HTTP {response.status_code}", _parse_retry_after(retry_after))
        response.raise_for_status()
        return response

    return with_retry(call, attempts=attempts, cap=cap)


def _parse_retry_after(value):
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _error(exc):
    message = f"{type(exc).__name__}: {exc}"
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key:
        for secret in {key, quote(key, safe="")}:
            message = message.replace(secret, "<redacted>")
    return f"ERROR: {message}"


def _reserve_arxiv_slot():
    """Keep at least 3 seconds between arXiv HTTP calls, including retries and parallel tool calls."""
    global _LAST_ARXIV_CALL
    with _ARXIV_LOCK:
        wait = 3.0 - (time.monotonic() - _LAST_ARXIV_CALL)
        if wait > 0:
            time.sleep(wait)
        _LAST_ARXIV_CALL = time.monotonic()


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    global _LAST_ARXIV_CALL
    terms = re.findall(r"[A-Za-z0-9-]+", query or "")
    if not terms:
        return "NO RESULTS"
    max_results = _clamp(max_results, 1, 30, 10)
    params = {
        "search_query": " AND ".join(f"all:{term}" for term in terms[:8]),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
        "start": 0,
    }
    try:
        response = _retryable_get(
            ARXIV_URL, params=params, timeout=30, attempts=7, cap=60, before_request=_reserve_arxiv_slot
        )
        root = xml.etree.ElementTree.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", ns):
            raw_id = (entry.findtext("atom:id", default="", namespaces=ns).rsplit("/", 1)[-1])
            paper_id = re.sub(r"v\d+$", "", raw_id)
            if not paper_id:
                continue
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10],
                "title": _clean(entry.findtext("atom:title", default="", namespaces=ns)),
                "summary": _clean(entry.findtext("atom:summary", default="", namespaces=ns), 600),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001 - tools must never raise to the agent
        return _error(exc)


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    params = {"limit": _clamp(limit, 1, 100, 30)}
    if date:
        params["date"] = date
    try:
        items = _retryable_get(HF_DAILY_URL, params=params, timeout=30).json()
        records = _hf_records(items, source_summary="summary")
        if keyword:
            needles = [part.lower() for part in re.findall(r"[A-Za-z0-9-]+", keyword)]
            records = [
                record for record in records
                if all(part in f"{record['title']} {record['summary']}".lower() for part in needles)
            ]
        records.sort(key=lambda r: r.get("upvotes") or 0, reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    if not (query or "").strip():
        return "NO RESULTS"
    try:
        items = _retryable_get(
            HF_SEARCH_URL,
            params={"q": query, "limit": _clamp(limit, 1, 50, 10)},
            timeout=30,
        ).json()
        records = _hf_records(items, source_summary="ai_summary")
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


def _hf_records(items, *, source_summary):
    records = []
    for item in items if isinstance(items, list) else []:
        paper = item.get("paper") if isinstance(item, dict) else None
        if not isinstance(paper, dict):
            paper = item if isinstance(item, dict) else {}
        paper_id = paper.get("id")
        if not paper_id:
            continue
        summary = paper.get(source_summary) or paper.get("ai_summary") or paper.get("summary") or item.get("summary", "")
        records.append({
            "id": paper_id,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": (paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clean(paper.get("title") or item.get("title", "")),
            "summary": _clean(summary, 600),
            "upvotes": paper.get("upvotes") or item.get("upvotes") or 0,
            "github": paper.get("githubRepo") or item.get("githubRepo") or "",
            "stars": paper.get("githubStars") or item.get("githubStars") or 0,
        })
    return records


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    if not (query or "").strip():
        return "NO RESULTS"
    try:
        text = _exa_call("web_search_exa", {
            "query": query,
            "objective": objective or f"Find reliable research sources about {query}",
            "numResults": _clamp(num_results, 1, 10, 5),
        })
        return text if text.strip() else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    if not (url or "").strip():
        return "NO RESULTS"
    try:
        text = _exa_call("web_fetch_exa", {"urls": [url]})
        text = text[:12000]
        return text if text.strip() else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


def _exa_endpoint():
    key = os.getenv("EXA_API_KEY", "").strip()
    if not key:
        return EXA_URL
    return f"{EXA_URL}?{urlencode({'exaApiKey': key})}"


def _exa_call(name, arguments):
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}

    def call():
        try:
            response = httpx.post(
                _exa_endpoint(),
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                json=payload,
                timeout=60,
            )
        except httpx.TransportError as exc:
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code in RETRYABLE_STATUS:
            raise RetryableError(f"HTTP {response.status_code}", _parse_retry_after(response.headers.get("Retry-After")))
        response.raise_for_status()
        data = _parse_sse_json(response.text)
        if "error" in data:
            raise RuntimeError(data["error"])
        result = data.get("result") or {}
        meta = result.get("_meta") or {}
        content_text = "\n".join(
            part.get("text", "") for part in result.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if _exa_rate_limited(meta, content_text):
            raise RetryableError("Exa rate limited")
        if result.get("isError"):
            raise RuntimeError(content_text or "Exa tool error")
        return content_text

    return with_retry(call, attempts=7, base=2.0, cap=60.0)


def _exa_rate_limited(meta, content_text):
    """Exa's free tier returns HTTP 200 plus a flag in result._meta, not HTTP 429.

    A normal rate-limit counter (requests remaining) is not a failure. Retry only when the
    payload says the limit was actually hit.
    """
    blob = json.dumps(meta, default=str).lower()
    if any(marker in blob for marker in ("exceeded", "too many requests", "rate limit reached", "rate_limited")):
        return True
    head = content_text[:800].lower()
    return any(cue in head for cue in ("rate limit", "too many requests", "exceeded your", "quota exceeded"))


def _parse_sse_json(text):
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                return json.loads(payload)
    return json.loads(text)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
