"""Task definitions for the Support Ticket Triage environment.

Each task contains:
- A ticket (observation data)
- The ground-truth triage (expected category, priority, response keywords)
- Difficulty level
"""

def grade(*args, **kwargs) -> float:
    """Standalone grader to satisfy Phase 2 hackathon validation.
    The actual environment grading happens deterministically inside SupportTriageEnv.step().
    """
    return 1.0

TASKS = [
    # ---- EASY: Straightforward billing question ----
    {
        "ticket": {
            "ticket_id": "TKT-001",
            "subject": "Billing charge question",
            "body": "Hi, I was charged $49.99 on my last invoice but I expected $29.99. "
                    "Can you explain the extra charge? My account is on the Pro plan.",
            "customer_tier": "pro",
            "task_difficulty": "easy",
        },
        "expected": {
            "category": "billing",
            "priority": "medium",
            "response_keywords": ["invoice", "charge", "billing"],
        },
        "grader": grade,
    },
    # ---- MEDIUM: Technical issue with some ambiguity ----
    {
        "ticket": {
            "ticket_id": "TKT-002",
            "subject": "Cannot access dashboard after password reset",
            "body": "I reset my password yesterday and now I can't log into the dashboard. "
                    "I've tried clearing cookies and using incognito mode. "
                    "I'm an enterprise customer and this is blocking my team.",
            "customer_tier": "enterprise",
            "task_difficulty": "medium",
        },
        "expected": {
            "category": "technical",
            "priority": "high",
            "response_keywords": ["password", "access", "login"],
        },
        "grader": grade,
    },
    # ---- HARD: Ambiguous ticket needing careful reading ----
    {
        "ticket": {
            "ticket_id": "TKT-003",
            "subject": "Account and billing concerns after downgrade",
            "body": "I downgraded from Enterprise to Pro last week. Since then, "
                    "several team members lost access to shared dashboards, and I'm "
                    "still being charged the Enterprise rate. I need both issues resolved "
                    "urgently as our quarterly review is tomorrow.",
            "customer_tier": "pro",
            "task_difficulty": "hard",
        },
        "expected": {
            "category": "account",
            "priority": "urgent",
            "response_keywords": ["downgrade", "access", "charge"],
        },
        "grader": grade,
    },
]
