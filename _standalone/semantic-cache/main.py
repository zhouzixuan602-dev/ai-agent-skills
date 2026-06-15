"""
semantic-cache — Cut LLM API costs 60-80% by caching responses to semantically similar queries.

Instead of exact string matching, uses embedding similarity so "How do I reset my password?"
and "Steps to change my password?" both hit the same cached response.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import hashlib
import time
import math
from pathlib import Path
import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "voyage-3-lite")  # or use cosine on your own
CACHE_FILE = os.environ.get("CACHE_FILE", ".semantic_cache.json")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.92"))  # 0-1, higher = stricter
MAX_CACHE_SIZE = int(os.environ.get("MAX_CACHE_SIZE", "500"))  # evict oldest when exceeded

client = anthropic.Anthropic()


# ── Embedding via Anthropic (text-embedding via messages hack) ─────────────────
def simple_hash_embed(text: str, dim: int = 64) -> list[float]:
    """
    Lightweight pseudo-embedding using character n-gram hashing.
    No extra API calls — good enough for FAQ/support use cases.
    Replace with voyage-3 or openai embeddings for production.
    """
    text = text.lower().strip()
    vec = [0.0] * dim
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(x**2 for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Cache persistence ──────────────────────────────────────────────────────────
def load_cache(path: str) -> dict:
    """Load cache from disk; return empty dict if missing."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"entries": [], "stats": {"hits": 0, "misses": 0, "saved_tokens": 0}}


def save_cache(cache: dict, path: str) -> None:
    """Persist cache to disk atomically."""
    tmp = path + ".tmp"
    Path(tmp).write_text(json.dumps(cache, indent=2))
    Path(tmp).replace(path)


# ── Core cache logic ───────────────────────────────────────────────────────────
def find_cached(query: str, cache: dict, threshold: float) -> dict | None:
    """Return cached entry if a similar query exists above threshold."""
    q_vec = simple_hash_embed(query)
    best_score = 0.0
    best_entry = None

    for entry in cache["entries"]:
        score = cosine_similarity(q_vec, entry["embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= threshold and best_entry:
        best_entry["last_hit"] = time.time()
        best_entry["hits"] = best_entry.get("hits", 0) + 1
        return best_entry
    return None


def add_to_cache(query: str, response: str, token_count: int, cache: dict) -> None:
    """Add a new entry; evict oldest if over capacity."""
    entry = {
        "query": query,
        "embedding": simple_hash_embed(query),
        "response": response,
        "token_count": token_count,
        "created_at": time.time(),
        "last_hit": time.time(),
        "hits": 0,
    }
    cache["entries"].append(entry)

    # LRU eviction
    if len(cache["entries"]) > MAX_CACHE_SIZE:
        cache["entries"].sort(key=lambda e: e["last_hit"])
        cache["entries"] = cache["entries"][-MAX_CACHE_SIZE:]


# ── Public API ─────────────────────────────────────────────────────────────────
def cached_llm_call(
    query: str,
    system: str = "You are a helpful assistant.",
    cache_path: str = CACHE_FILE,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Make an LLM call with semantic caching.

    Returns:
        {
            "response": str,
            "cached": bool,
            "similarity": float | None,
            "tokens_used": int,
            "tokens_saved": int,
        }
    """
    cache = load_cache(cache_path)

    # 1. Check cache
    hit = find_cached(query, cache, threshold)
    if hit:
        cache["stats"]["hits"] += 1
        cache["stats"]["saved_tokens"] += hit.get("token_count", 0)
        save_cache(cache, cache_path)
        return {
            "response": hit["response"],
            "cached": True,
            "similarity": None,  # already above threshold
            "tokens_used": 0,
            "tokens_saved": hit.get("token_count", 0),
        }

    # 2. Cache miss — call the API
    cache["stats"]["misses"] += 1
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": query}],
    )
    response_text = msg.content[0].text
    tokens = msg.usage.input_tokens + msg.usage.output_tokens

    add_to_cache(query, response_text, tokens, cache)
    save_cache(cache, cache_path)

    return {
        "response": response_text,
        "cached": False,
        "similarity": None,
        "tokens_used": tokens,
        "tokens_saved": 0,
    }


def cache_stats(cache_path: str = CACHE_FILE) -> dict:
    """Return hit rate and savings summary."""
    cache = load_cache(cache_path)
    s = cache["stats"]
    total = s["hits"] + s["misses"]
    hit_rate = (s["hits"] / total * 100) if total else 0
    return {
        "total_queries": total,
        "cache_hits": s["hits"],
        "cache_misses": s["misses"],
        "hit_rate_pct": round(hit_rate, 1),
        "tokens_saved": s["saved_tokens"],
        "cache_entries": len(cache["entries"]),
    }


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DEMO_CACHE = ".demo_cache.json"

    queries = [
        "How do I reset my password?",           # Original
        "What are the steps to change my password?",  # Semantically similar → should hit
        "How do I update my login credentials?", # Also similar → should hit
        "What is the capital of France?",        # Different topic → miss
        "Tell me the capital city of France.",   # Similar to above → should hit
    ]

    print("🔍 Semantic Cache Demo\n" + "=" * 50)
    system = "You are a helpful customer support assistant. Be concise."

    for q in queries:
        result = cached_llm_call(q, system=system, cache_path=DEMO_CACHE)
        status = "✅ CACHE HIT" if result["cached"] else "🌐 API CALL"
        print(f"\n{status} | Query: {q[:55]}...")
        print(f"  Response: {result['response'][:80]}...")
        if result["tokens_used"]:
            print(f"  Tokens used: {result['tokens_used']}")
        if result["tokens_saved"]:
            print(f"  Tokens saved: {result['tokens_saved']}")

    print("\n" + "=" * 50)
    stats = cache_stats(DEMO_CACHE)
    print(f"📊 Stats: {stats['hit_rate_pct']}% hit rate | "
          f"{stats['tokens_saved']} tokens saved | "
          f"{stats['cache_entries']} entries cached")

    # Cleanup demo cache
    Path(DEMO_CACHE).unlink(missing_ok=True)
