"""
precompute.py — Run ONCE before rank.py.

Loads all 100K candidates from candidates.jsonl (or .jsonl.gz),
builds a text representation for each, embeds them with
all-MiniLM-L6-v2 (CPU, ~2-3 min), and saves:
  - embeddings.npy   : float32 array, shape (N, 384)
  - candidate_ids.npy: string array, shape (N,)

Usage:
    python precompute.py --candidates candidates.jsonl
    python precompute.py --candidates candidates.jsonl.gz   # gzip works too
"""

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ── text builder ───────────────────────────────────────────────────────────────

def build_candidate_text(c: dict) -> str:
    """
    Concatenate the most signal-rich fields into a single string for embedding.
    Order matters: title and summary first (highest signal), then skills and
    career descriptions.
    """
    parts = []
    p = c.get("profile", {})

    # 1. Title + headline
    if p.get("current_title"):
        parts.append(p["current_title"])
    if p.get("headline"):
        parts.append(p["headline"])

    # 2. Profile summary (the best semantic signal)
    if p.get("summary"):
        parts.append(p["summary"])

    # 3. Skills (name + proficiency — keeps text short but informative)
    skills = c.get("skills", [])
    if skills:
        skill_str = ", ".join(
            f"{s['name']} ({s.get('proficiency', '')})"
            for s in skills[:20]  # cap at 20 to avoid token explosion
        )
        parts.append("Skills: " + skill_str)

    # 4. Career descriptions (most recent 3 jobs)
    for job in c.get("career_history", [])[:3]:
        title = job.get("title", "")
        company = job.get("company", "")
        desc = job.get("description", "") or ""
        if title or desc:
            entry = f"{title} at {company}: {desc[:300]}"  # trim long descriptions
            parts.append(entry)

    return " | ".join(parts)


# ── loader ─────────────────────────────────────────────────────────────────────

def load_candidates(path: str):
    """Yields candidate dicts from .jsonl or .jsonl.gz"""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt"
    with opener(p, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out-dir", default=".",
                        help="Where to save embeddings.npy and candidate_ids.npy")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "embeddings.npy"
    ids_path = out_dir / "candidate_ids.npy"

    print(f"Loading candidates from {args.candidates}...")
    t0 = time.time()

    candidates = list(load_candidates(args.candidates))
    print(f"Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    print(f"\nBuilding text representations...")
    texts = [build_candidate_text(c) for c in tqdm(candidates)]
    ids   = np.array([c["candidate_id"] for c in candidates], dtype="U20")

    print(f"\nLoading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"\nEmbedding {len(texts):,} candidates (CPU, batch={args.batch_size})...")
    t1 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize so cosine sim = dot product
    )
    elapsed = time.time() - t1
    print(f"Embedding done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print(f"\nSaving to {out_dir}/...")
    np.save(emb_path, embeddings.astype(np.float32))
    np.save(ids_path, ids)
    print(f"  embeddings.npy : {embeddings.shape}  {emb_path.stat().st_size/1e6:.1f} MB")
    print(f"  candidate_ids.npy : {ids.shape}")
    print("\nDone. Run rank.py next.")


if __name__ == "__main__":
    main()