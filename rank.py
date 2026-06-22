"""
rank.py — The timed ranking step (≤5 min, CPU, no network).

Loads precomputed embeddings, embeds the JD, computes cosine similarity,
shortlists top-1000, applies career + behavioral scoring, outputs top-100 CSV.

Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv
    python rank.py --candidates candidates.jsonl.gz --out team_xxx.csv
    python rank.py --candidates candidates.jsonl --out submission.csv --emb-dir ./precomputed
"""

import argparse
import csv
import datetime
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from score_features import career_fit_score, behavioral_score
from detect_honeypots import honeypot_penalty

# ── Scoring weights ────────────────────────────────────────────────────────────
# These add up to 1.0. Tweak based on your validation experiments.
W_SEMANTIC  = 0.40
W_CAREER    = 0.30
W_BEHAVIORAL= 0.20
W_HONEYPOT  = 0.10   # honeypot_penalty is a multiplier; see below

# Top-K candidates to run full feature scoring on (semantic pre-filter)
SHORTLIST_K = 1000

# ── JD text ───────────────────────────────────────────────────────────────────
# Carefully written to capture what the JD *means*, not just its keywords.
# Focus: production embeddings + vector DB, product company, shipper mindset.
JD_TEXT = """
Senior AI Engineer role at Redrob AI, a Series A AI-native talent intelligence platform.
Founding team hire to own the intelligence layer: ranking, retrieval, and matching systems.

Required: Production experience with embeddings-based retrieval systems using sentence-transformers,
OpenAI embeddings, BGE, E5, or similar models. Deployed to real users. Handled embedding drift,
index refresh, retrieval-quality regression in production.

Required: Production experience with vector databases or hybrid search infrastructure: Pinecone,
Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. Operational experience matters, not just tutorials.

Required: Strong Python. Production-quality code.

Required: Designed evaluation frameworks for ranking systems: NDCG, MRR, MAP, offline-to-online
correlation, A/B test interpretation.

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost or neural),
HR-tech or marketplace experience, distributed systems.

NOT wanted: Title-chasers. Framework tutorial writers. Consulting-only career (TCS, Infosys,
Wipro, Accenture). Pure computer vision or speech experts without NLP/IR background.
People who moved to architecture roles and stopped writing code.

Ideal: 6-8 years total, 4-5 in applied ML at product companies. Shipped at least one end-to-end
ranking, search, or recommendation system at scale. Strong opinions on retrieval, evaluation,
LLM integration backed by systems they actually built.

Location: Pune or Noida preferred, open to Hyderabad, Mumbai, Delhi NCR. India only. Hybrid.
Notice period: sub-30 days preferred.
"""


# ── helpers ────────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> dict:
    """Returns dict: candidate_id -> candidate dict"""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    candidates = {}
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                candidates[c["candidate_id"]] = c
    return candidates


def generate_reasoning(candidate: dict, sem_score: float,
                       career_s: float, behav_s: float,
                       hp: float, final_score: float) -> str:
    """
    Generate a specific, factual 1-2 sentence reasoning for the CSV.
    Uses only data that actually exists in the profile — no hallucination.
    """
    p = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])

    title = p.get("current_title", "unknown title")
    yoe = p.get("years_of_experience", 0) or 0
    company = p.get("current_company", "")
    location = p.get("location", "")
    country = p.get("country", "")

    # Key skills for this role
    core_keywords = {
        "qdrant", "pinecone", "weaviate", "faiss", "elasticsearch",
        "opensearch", "milvus", "embeddings", "semantic search",
        "retrieval", "rag", "ranking", "vector", "sentence-transformers",
        "fine-tuning", "fine tuning", "lora", "ndcg", "mrr",
        "information retrieval", "hybrid search", "bm25",
    }
    matched_skills = [
        s["name"] for s in skills
        if any(kw in s["name"].lower() for kw in core_keywords)
    ][:4]

    # Recency
    last_active_raw = sig.get("last_active_date")
    recency_str = ""
    if last_active_raw:
        try:
            last_active = datetime.date.fromisoformat(last_active_raw)
            days_ago = (datetime.date.today() - last_active).days
            recency_str = f"active {days_ago}d ago"
        except ValueError:
            pass

    response_rate = sig.get("recruiter_response_rate") or 0.0
    notice = sig.get("notice_period_days")
    open_to_work = sig.get("open_to_work_flag", False)

    # Build sentence 1: profile summary
    loc_display = f"{location}, {country}" if location and country else (location or country)
    sentence1_parts = [f"{yoe:.0f} yrs exp"]
    if company:
        sentence1_parts.append(f"currently at {company}")
    if loc_display:
        sentence1_parts.append(f"based in {loc_display}")
    if matched_skills:
        sentence1_parts.append(f"relevant skills: {', '.join(matched_skills)}")
    sentence1 = f"{title}; {'; '.join(sentence1_parts)}."

    # Build sentence 2: signals + concerns
    signals_parts = []
    if open_to_work:
        signals_parts.append("open to work")
    if recency_str:
        signals_parts.append(recency_str)
    if response_rate >= 0.6:
        signals_parts.append(f"{response_rate:.0%} response rate")
    elif response_rate > 0 and response_rate < 0.4:
        signals_parts.append(f"low response rate ({response_rate:.0%})")
    if notice is not None:
        signals_parts.append(f"{notice}d notice")

    # Concerns
    concerns = []
    if hp < 0.5:
        concerns.append("profile consistency issues flagged")
    if yoe < 4:
        concerns.append(f"below preferred experience range ({yoe:.1f} yrs)")
    if yoe > 12:
        concerns.append("over-experienced for stated range")
    if career_s < 0.3:
        concerns.append("limited product-company AI experience")

    sentence2 = ""
    if signals_parts:
        sentence2 = "Signals: " + ", ".join(signals_parts) + "."
    if concerns:
        sentence2 += (" " if sentence2 else "") + "Concerns: " + "; ".join(concerns) + "."

    reasoning = sentence1
    if sentence2:
        reasoning += " " + sentence2
    return reasoning.strip()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob hackathon")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", required=True,
                        help="Output CSV path (e.g. team_xxx.csv)")
    parser.add_argument("--emb-dir", default=".",
                        help="Directory containing embeddings.npy and candidate_ids.npy")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Must match the model used in precompute.py")
    args = parser.parse_args()

    wall_start = time.time()

    emb_path = Path(args.emb_dir) / "embeddings.npy"
    ids_path = Path(args.emb_dir) / "candidate_ids.npy"

    if not emb_path.exists() or not ids_path.exists():
        print("ERROR: embeddings.npy / candidate_ids.npy not found.")
        print("Run precompute.py first:")
        print("  python precompute.py --candidates candidates.jsonl")
        raise SystemExit(1)

    # ── Step 1: Load precomputed embeddings ────────────────────────────────────
    print("[1/5] Loading precomputed embeddings...")
    t = time.time()
    embeddings = np.load(emb_path)          # (N, 384) float32
    cand_ids   = np.load(ids_path)          # (N,) str
    print(f"      {len(cand_ids):,} candidates  ({time.time()-t:.1f}s)")

    # ── Step 2: Embed the JD ───────────────────────────────────────────────────
    print("[2/5] Embedding JD...")
    t = time.time()
    model = SentenceTransformer(args.model)
    jd_vec = model.encode(
        JD_TEXT.strip(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    print(f"      Done ({time.time()-t:.1f}s)")

    # ── Step 3: Cosine similarity (dot product since both are LS2-normalized) ──
    print(f"[3/5] Computing cosine similarity for {len(cand_ids):,} candidates...")
    t = time.time()
    scores_semantic = embeddings @ jd_vec   # shape (N,)
    print(f"      Done ({time.time()-t:.1f}s)")

    # Top-SHORTLIST_K semantic candidates for deep scoring
    #SHORTLIST_K = min(1000, len(scores_semantic))

    top_idx = np.argpartition(scores_semantic, -SHORTLIST_K)[-SHORTLIST_K:]
    top_idx = top_idx[np.argsort(scores_semantic[top_idx])[::-1]]   # sorted desc
    print(f"      Shortlisted top {SHORTLIST_K}")

    # ── Step 4: Load candidate profiles for shortlist ─────────────────────────
    print(f"[4/5] Loading candidate profiles from {args.candidates}...")
    t = time.time()
    all_candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(all_candidates):,} profiles ({time.time()-t:.1f}s)")

    # ── Step 5: Feature scoring on shortlist ─────────────────────────────────
    print(f"[5/5] Scoring top {SHORTLIST_K} candidates...")
    t = time.time()

    results = []
    for i in top_idx:
        cid = cand_ids[i]
        sem = float(scores_semantic[i])

        c = all_candidates.get(cid)
        if c is None:
            continue

        c_score  = career_fit_score(c)
        b_score  = behavioral_score(c)
        hp       = honeypot_penalty(c)

        # Weighted combination
        raw_score = (
            W_SEMANTIC   * sem +
            W_CAREER     * c_score +
            W_BEHAVIORAL * b_score
        )
        # Apply honeypot penalty as a multiplier
        # hp = 1.0 → no change; hp = 0.1 → nearly eliminated
        final = raw_score * hp

        results.append({
            "candidate_id": cid,
            "sem":   sem,
            "career": c_score,
            "behav":  b_score,
            "hp":     hp,
            "score":  final,
            "candidate": c,
        })

    # Sort by final score descending
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    print(f"      Done ({time.time()-t:.1f}s)")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(results[:100], start=1):
            reasoning = generate_reasoning(
                r["candidate"], r["sem"], r["career"],
                r["behav"], r["hp"], r["score"]
            )
            writer.writerow([
                r["candidate_id"],
                rank,
                f"{r['score']:.6f}",
                reasoning,
            ])

    elapsed_total = time.time() - wall_start
    print(f"\nDone in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"Output: {out_path}")
    print(f"Top-5 candidates:")
    for i, r in enumerate(results[:5], 1):
        p = r["candidate"]["profile"]
        print(f"  {i}. {r['candidate_id']} | {p.get('current_title')} | "
              f"score={r['score']:.4f} "
              f"(sem={r['sem']:.3f} career={r['career']:.3f} "
              f"behav={r['behav']:.3f} hp={r['hp']:.3f})")


if __name__ == "__main__":
    main()