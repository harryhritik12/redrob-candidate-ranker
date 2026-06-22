"""
Feature scoring for the Redrob Senior AI Engineer JD.

Two scores:
  career_fit_score(candidate)   -> float 0.0-1.0
  behavioral_score(candidate)   -> float 0.0-1.0

Both are called by rank.py on the top-1000 semantic shortlist.
"""

import datetime
from typing import Dict, Any

# ── JD-specific constants ──────────────────────────────────────────────────────

# Companies the JD explicitly says it does NOT want (consulting-only career)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree",
}

# Skills that signal genuine AI/ML retrieval experience (not just keywords)
CORE_SKILL_KEYWORDS = {
    "embeddings", "vector database", "vector search", "semantic search",
    "retrieval", "rag", "ranking", "faiss", "qdrant", "pinecone",
    "weaviate", "milvus", "opensearch", "elasticsearch",
    "sentence-transformers", "sentence transformers", "dense retrieval",
    "hybrid search", "bm25", "information retrieval",
    "ndcg", "mrr", "learning to rank", "reranking", "re-ranking",
    "fine-tuning", "fine tuning", "lora", "qlora",
}

# Preferred India locations from the JD
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bengaluru",
    "bangalore", "gurugram", "gurgaon", "ncr", "india",
}

# Industries that count as "product company" experience
PRODUCT_INDUSTRIES = {
    "technology", "software", "saas", "fintech", "edtech", "healthtech",
    "ecommerce", "e-commerce", "marketplace", "ai", "machine learning",
    "internet", "media", "gaming", "telecommunications",
}

# JD says explicitly: no pure CV/speech/robotics experts
WRONG_DOMAIN_TITLES = {
    "computer vision", "speech recognition", "robotics", "object detection",
    "image segmentation", "autonomous driving",
}

TODAY = datetime.date.today()


# ── helpers ────────────────────────────────────────────────────────────────────

def _company_is_consulting(company_name: str) -> bool:
    name = company_name.lower()
    return any(firm in name for firm in CONSULTING_FIRMS)


def _career_is_consulting_only(career_history: list) -> bool:
    """True if every job in history is at a known consulting firm."""
    if not career_history:
        return False
    return all(_company_is_consulting(j.get("company", "")) for j in career_history)


def _has_product_company_exp(career_history: list) -> bool:
    """True if candidate has at least one role at a non-consulting company."""
    for job in career_history:
        if not _company_is_consulting(job.get("company", "")):
            industry = (job.get("industry") or "").lower()
            size = job.get("company_size", "")
            # Any non-consulting company with tech/product industry or medium+ size
            if any(ind in industry for ind in PRODUCT_INDUSTRIES):
                return True
            if size in ("51-200", "201-500", "501-1000", "1001-5000",
                        "5001-10000", "10001+"):
                return True
    return False


def _count_core_skills(skills: list) -> int:
    """Count skills that map to what the JD actually cares about."""
    count = 0
    for s in skills:
        name = s.get("name", "").lower()
        if any(kw in name for kw in CORE_SKILL_KEYWORDS):
            count += 1
    return count


def _has_core_skill_in_career(career_history: list) -> bool:
    """Check job descriptions mention actual retrieval/ML work (not just listed skills)."""
    for job in career_history:
        desc = (job.get("description") or "").lower()
        matches = sum(1 for kw in CORE_SKILL_KEYWORDS if kw in desc)
        if matches >= 2:
            return True
    return False


def _location_score(profile: dict) -> float:
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        return 1.0
    if "india" in country:
        return 0.8
    # Willing to relocate is handled via signals
    return 0.2


def _yoe_score(yoe: float) -> float:
    """JD says 5-9 yrs, but 4 is OK. <4 or >15 are weak."""
    if yoe is None:
        return 0.3
    if 4 <= yoe <= 9:
        return 1.0
    if 9 < yoe <= 12:
        return 0.7
    if 3 <= yoe < 4:
        return 0.6
    if 12 < yoe <= 15:
        return 0.5
    return 0.2  # <3 or >15


def _title_relevance(title: str) -> float:
    title = title.lower()
    if any(t in title for t in [
        "ai engineer", "ml engineer", "machine learning engineer",
        "applied scientist", "nlp engineer", "search engineer",
        "data scientist", "research engineer", "senior engineer",
        "principal engineer", "staff engineer",
    ]):
        return 1.0
    if any(t in title for t in [
        "software engineer", "backend engineer", "platform engineer",
        "data engineer", "full stack", "fullstack",
    ]):
        return 0.6
    if any(t in title for t in WRONG_DOMAIN_TITLES):
        return 0.2
    return 0.3


def _current_company_not_consulting(profile: dict) -> float:
    company = (profile.get("current_company") or "").lower()
    if _company_is_consulting(company):
        return 0.4
    return 1.0


# ── main scoring functions ─────────────────────────────────────────────────────

def career_fit_score(candidate: Dict[str, Any]) -> float:
    """
    Score 0.0-1.0 based on career history, skills, title, location.
    This is the 'smart recruiter' layer — catches good candidates without
    AI keywords and penalises keyword stuffers with wrong career trajectory.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Hard disqualifiers → very low score (not 0 to avoid sorting artifacts)
    if _career_is_consulting_only(career):
        return 0.05

    yoe = profile.get("years_of_experience") or 0.0

    # Component scores
    yoe_s       = _yoe_score(yoe)
    title_s     = _title_relevance(profile.get("current_title", ""))
    loc_s       = _location_score(profile)
    product_s   = 1.0 if _has_product_company_exp(career) else 0.2
    core_skill_count = _count_core_skills(skills)
    skill_s     = min(1.0, core_skill_count / 4.0)   # 4+ core skills = max
    career_ml_s = 1.0 if _has_core_skill_in_career(career) else 0.3
    no_consult_s = _current_company_not_consulting(profile)

    # Wrong domain penalty
    title_lower = profile.get("current_title", "").lower()
    wrong_domain = any(wd in title_lower for wd in WRONG_DOMAIN_TITLES)
    if wrong_domain:
        return 0.1

    score = (
        0.20 * yoe_s +
        0.18 * title_s +
        0.15 * product_s +
        0.17 * skill_s +
        0.15 * career_ml_s +
        0.10 * loc_s +
        0.05 * no_consult_s
    )
    return round(min(1.0, max(0.0, score)), 6)


def behavioral_score(candidate: Dict[str, Any]) -> float:
    """
    Score 0.0-1.0 based on Redrob platform behavioral signals.
    A perfect-on-paper candidate who is inactive is, for hiring purposes,
    not actually available. This is a multiplier, not a replacement.
    """
    sig = candidate.get("redrob_signals", {})

    # ── Availability signals ──
    open_to_work = 1.0 if sig.get("open_to_work_flag") else 0.4

    # Recency of last login
    last_active_raw = sig.get("last_active_date")
    if last_active_raw:
        try:
            last_active = datetime.date.fromisoformat(last_active_raw)
            days_ago = (TODAY - last_active).days
            if days_ago <= 14:
                recency = 1.0
            elif days_ago <= 30:
                recency = 0.85
            elif days_ago <= 60:
                recency = 0.65
            elif days_ago <= 120:
                recency = 0.40
            else:
                recency = 0.15
        except ValueError:
            recency = 0.3
    else:
        recency = 0.3

    # Notice period — JD prefers sub-30d, can buy out 30d
    notice = sig.get("notice_period_days") or 90
    if notice <= 15:
        notice_s = 1.0
    elif notice <= 30:
        notice_s = 0.95
    elif notice <= 60:
        notice_s = 0.7
    elif notice <= 90:
        notice_s = 0.5
    else:
        notice_s = 0.25

    # ── Responsiveness signals ──
    response_rate = sig.get("recruiter_response_rate") or 0.0
    # response_rate is 0.0-1.0
    response_s = float(response_rate)

    avg_resp_h = sig.get("avg_response_time_hours") or 48.0
    if avg_resp_h <= 4:
        resp_time_s = 1.0
    elif avg_resp_h <= 12:
        resp_time_s = 0.85
    elif avg_resp_h <= 24:
        resp_time_s = 0.65
    elif avg_resp_h <= 48:
        resp_time_s = 0.45
    else:
        resp_time_s = 0.2

    interview_completion = sig.get("interview_completion_rate") or 0.0
    interview_s = float(interview_completion)

    # ── Engagement quality ──
    github = sig.get("github_activity_score") or -1
    if github == -1:
        github_s = 0.4   # no GitHub — not great for AI engineer role
    else:
        github_s = min(1.0, github / 80.0)

    # Willing to relocate gives a small boost
    relocate_s = 0.1 if sig.get("willing_to_relocate") else 0.0

    # Verified contact = reachable
    verified = 0.05 if (sig.get("verified_email") or sig.get("verified_phone")) else 0.0

    score = (
        0.20 * open_to_work +
        0.20 * recency +
        0.15 * notice_s +
        0.15 * response_s +
        0.10 * resp_time_s +
        0.10 * interview_s +
        0.07 * github_s +
        0.03 * relocate_s +
        0.00 * verified   # tiny boost, don't waste weight
    )
    score += verified
    return round(min(1.0, max(0.0, score)), 6)