"""
Honeypot detection for the Redrob challenge.

The dataset contains ~80 honeypot profiles with subtly impossible data.
If >10% of your top-100 are honeypots, the submission is disqualified.

honeypot_penalty(candidate) -> float
  Returns a penalty multiplier: 1.0 = clean, 0.0 = certain honeypot.
  Multiply your final score by this value.
"""

import datetime
from typing import Dict, Any

TODAY = datetime.date.today()

# Companies founded after certain years — any candidate claiming
# many years there is a honeypot flag. We use duration vs company age heuristic.
# (We can't know exact founding dates, so we check internal consistency instead.)

KNOWN_PROFICIENCY_LEVELS = {"beginner", "intermediate", "advanced", "expert"}


def _check_timeline_consistency(career_history: list, stated_yoe: float) -> float:
    """
    Check if stated YoE matches the sum of career history durations.
    A stated 8 yrs but only 3 yrs of actual history = suspicious.
    Returns penalty: 1.0 = consistent, 0.0 = impossible.
    """
    if not career_history:
        return 0.8  # no history at all is odd but not definitively a honeypot

    total_months = sum(j.get("duration_months") or 0 for j in career_history)
    total_years = total_months / 12.0

    if stated_yoe is None or stated_yoe <= 0:
        return 0.8

    ratio = total_years / stated_yoe
    # Should be close to 1.0 — allow gaps (study, travel, between jobs)
    if 0.5 <= ratio <= 1.6:
        return 1.0
    if 0.3 <= ratio < 0.5 or 1.6 < ratio <= 2.0:
        return 0.6
    # Extreme mismatch — e.g. 8 yrs stated but only 1 yr of history
    return 0.1


def _check_skill_plausibility(skills: list, stated_yoe: float) -> float:
    """
    An 'expert' in 10+ skills with 2 yrs experience is suspicious.
    Also check for skills where duration_months >> stated total career.
    """
    if not skills:
        return 1.0

    expert_count = sum(
        1 for s in skills
        if (s.get("proficiency") or "").lower() == "expert"
    )
    yoe = stated_yoe or 0

    # More than 8 expert skills is unrealistic regardless of yoe
    if expert_count > 8:
        return 0.3
    # Expert in >5 skills with <4 yrs experience
    if expert_count > 5 and yoe < 4:
        return 0.4

    # Check if any skill's duration_months exceeds total career by >20%
    total_career_months = (yoe or 0) * 12
    for s in skills:
        skill_months = s.get("duration_months") or 0
        if total_career_months > 0 and skill_months > total_career_months * 1.2:
            return 0.2  # used a skill longer than their career = impossible

    return 1.0


def _check_date_consistency(career_history: list) -> float:
    """
    Check for overlapping jobs that don't make sense, or future dates.
    Returns penalty 0.0-1.0.
    """
    if not career_history:
        return 1.0

    issues = 0
    for job in career_history:
        start_raw = job.get("start_date")
        end_raw = job.get("end_date")

        try:
            start = datetime.date.fromisoformat(start_raw) if start_raw else None
        except (ValueError, TypeError):
            issues += 1
            continue

        try:
            end = datetime.date.fromisoformat(end_raw) if end_raw else None
        except (ValueError, TypeError):
            end = None

        if start and start > TODAY:
            issues += 2  # start date in the future
        if end and end > TODAY:
            issues += 1  # end date in the future (minor — data could be stale)
        if start and end and end < start:
            issues += 3  # end before start = impossible

    if issues == 0:
        return 1.0
    if issues <= 1:
        return 0.7
    if issues <= 3:
        return 0.4
    return 0.1


def _check_profile_credibility(profile: dict, career_history: list) -> float:
    """
    Broad plausibility: does the title match the career?
    A 'Senior AI Engineer' whose entire history is in retail sales = suspicious.
    """
    title = (profile.get("current_title") or "").lower()
    yoe = profile.get("years_of_experience") or 0

    tech_title = any(t in title for t in [
        "engineer", "developer", "scientist", "analyst", "architect",
        "researcher", "technical", "ai", "ml", "data",
    ])

    if not tech_title:
        return 1.0  # Not claiming to be an engineer — no contradiction

    # If claims to be a senior/staff engineer, should have 4+ yrs
    senior_title = any(t in title for t in ["senior", "staff", "principal", "lead"])
    if senior_title and yoe < 3:
        return 0.3

    # Career history should have at least some tech roles
    if career_history:
        tech_jobs = sum(
            1 for j in career_history
            if any(t in (j.get("title") or "").lower() for t in [
                "engineer", "developer", "scientist", "analyst", "architect",
                "researcher", "technical",
            ])
        )
        if tech_jobs == 0:
            return 0.2  # claims tech title but no tech history

    return 1.0


# ── main function ──────────────────────────────────────────────────────────────

def honeypot_penalty(candidate: Dict[str, Any]) -> float:
    """
    Returns a multiplier 0.0-1.0 to apply to the final score.
    1.0 = no honeypot signal, 0.0 = definite honeypot.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    yoe = profile.get("years_of_experience") or 0.0

    timeline_ok   = _check_timeline_consistency(career, yoe)
    skill_ok      = _check_skill_plausibility(skills, yoe)
    dates_ok      = _check_date_consistency(career)
    credibility   = _check_profile_credibility(profile, career)

    # Combine — any single strong signal is enough to heavily penalise
    combined = (
        0.30 * timeline_ok +
        0.30 * skill_ok +
        0.20 * dates_ok +
        0.20 * credibility
    )

    # If any check is definitively bad (0.1 or below), cap the combined score
    min_check = min(timeline_ok, skill_ok, dates_ok, credibility)
    if min_check <= 0.1:
        combined = min(combined, 0.15)

    return round(combined, 6)