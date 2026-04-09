"""Task definitions for the Support Ticket Triage environment.

Each task contains:
- A ticket (observation data)
- The ground-truth triage (expected category, priority, response keywords)
- Difficulty level
"""

def _evaluate_action(action, expected) -> float:
    cat = action.get("category", "") if isinstance(action, dict) else getattr(action, "category", "")
    pri = action.get("priority", "") if isinstance(action, dict) else getattr(action, "priority", "")
    resp = action.get("response_snippet", "") if isinstance(action, dict) else getattr(action, "response_snippet", "")
    
    reward = 0.0
    if cat.lower() == expected["category"]: reward += 0.4
    if pri.lower() == expected["priority"]: reward += 0.3
    
    expected_keywords = expected["response_keywords"]
    snippet_lower = resp.lower()
    matched = sum(1 for kw in expected_keywords if kw in snippet_lower)
    reward += round(0.3 * (matched / max(1, len(expected_keywords))), 4)
    
    return round(reward, 4)

def grade_tkt_001(*args, **kwargs) -> float:
    action = kwargs.get("action") or kwargs.get("output") or (args[0] if args else {})
    return _evaluate_action(action, TASKS[0]["expected"])

def grade_tkt_002(*args, **kwargs) -> float:
    action = kwargs.get("action") or kwargs.get("output") or (args[0] if args else {})
    return _evaluate_action(action, TASKS[1]["expected"])

def grade_tkt_003(*args, **kwargs) -> float:
    action = kwargs.get("action") or kwargs.get("output") or (args[0] if args else {})
    return _evaluate_action(action, TASKS[2]["expected"])

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
        "grader": grade_tkt_001,
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
        "grader": grade_tkt_002,
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
        "grader": grade_tkt_003,
    },
]
