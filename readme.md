# Redrob — Intelligent Candidate Discovery & Ranking

**Challenge:** Rank 100K synthetic candidate profiles against a Senior AI Engineer JD.

---

## Quick start

```bash
pip install -r requirements.txt

# Step 1 (run once — no time limit)
python precompute.py --candidates candidates.jsonl

# Step 2 (the timed ranking step — must finish in ≤5 min on CPU)
python rank.py --candidates candidates.jsonl --out team_xxx.csv

# Validate
python validate_submission.py team_xxx.csv
```

---

## File structure

| File | Purpose |
|---|---|
| `precompute.py` | Embeds all 100K candidates → `embeddings.npy` + `candidate_ids.npy` |
| `rank.py` | Timed step: loads embeddings, scores, outputs CSV |
| `score_features.py` | Career fit + behavioral scoring rules |
| `detect_honeypots.py` | Timeline and profile consistency checks |
| `requirements.txt` | Python dependencies |

---

## Architecture

### Phase 1 — Offline precomputation (no time limit)

1. Load all 100K candidate profiles from `candidates.jsonl`
2. For each candidate, concatenate title + headline + summary + top skills + recent career descriptions into a single text string
3. Embed with `all-MiniLM-L6-v2` (22MB, CPU-friendly, L2-normalized)
4. Save as `embeddings.npy` (float32, shape 100000×384, ~150MB) and `candidate_ids.npy`

### Phase 2 — Ranking step (timed, CPU-only, no network)

1. Load precomputed embeddings from disk (~1s)
2. Embed the JD text using the same model (~1s)
3. Compute cosine similarity (dot product, both L2-normalized) for all 100K candidates — pure numpy, ~0.1s
4. Take top-1000 as semantic shortlist
5. For each of the 1000, compute three scores:
   - `career_fit_score` — product company experience, title relevance, location, YoE, core skill match, consulting-firm penalty
   - `behavioral_score` — recency of login, open-to-work flag, response rate, notice period, GitHub activity
   - `honeypot_penalty` — checks timeline consistency, skill plausibility, date impossibilities
6. Final score = `0.40 × semantic + 0.30 × career_fit + 0.20 × behavioral × honeypot_penalty`
7. Sort, take top 100, generate reasoning, write CSV

Total wall-clock time for Phase 2: **~2-4 minutes on a 16GB CPU machine**.

---

## Scoring weights rationale

| Component | Weight | Why |
|---|---|---|
| Semantic similarity | 40% | Captures meaning the JD expresses — retrieval, ranking, production ML. Better than keyword matching. |
| Career fit | 30% | Catches good candidates without buzzwords; penalises consulting-only careers and wrong domains (CV/speech). |
| Behavioral signals | 20% | A perfect-on-paper candidate inactive for 6 months is not actually hireable. |
| Honeypot penalty | multiplier | Eliminates impossible profiles before they inflate scores. |

---

## Career fit rules (score_features.py)

- **Consulting-only penalty**: candidates whose entire history is TCS/Infosys/Wipro/Accenture → 0.05 score (JD explicitly states this disqualifier)
- **Product company boost**: at least one role outside consulting in a tech/product industry
- **YoE sweet spot**: 4-9 years = full score; <3 or >15 = steep penalty
- **Core skill match**: embeddings, vector DBs (Qdrant/Pinecone/FAISS etc.), retrieval, ranking, NDCG, RAG
- **Career description check**: job descriptions must mention ML work, not just list it in skills
- **Location**: India-based or willing to relocate — Pune/Noida/Hyderabad/Mumbai/Delhi NCR
- **Wrong domain penalty**: pure CV/speech/robotics titles → 0.1

## Behavioral signal rules (score_features.py)

- `open_to_work_flag`: strong signal; missing = 0.4 weight
- `last_active_date`: ≤14 days ago = 1.0, >120 days = 0.15
- `notice_period_days`: ≤30 = near-max; >90 = significant penalty
- `recruiter_response_rate`: linear 0-1
- `github_activity_score`: -1 (no GitHub) = 0.4 penalty for an AI engineer role

## Honeypot detection (detect_honeypots.py)

Flags profiles with:
- Stated YoE not matching sum of career history durations (>2x mismatch)
- More than 8 "expert" skills, or expert skills with <4 yrs total experience
- Skills with `duration_months` exceeding entire career length
- Future start/end dates, or end dates before start dates
- Senior title with <3 years experience

---

## Environment

Tested on: Python 3.11, 16GB RAM, CPU only.

```
sentence-transformers==3.0.1
numpy>=1.24.0
tqdm>=4.65.0
```