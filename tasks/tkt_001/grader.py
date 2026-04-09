import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
# In server environments, use API_KEY first for proxy compatibility
API_KEY = os.getenv("API_KEY", os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here"))

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _client

def create_grader_prompt(ticket, expected, action) -> str:
    # Safely extract action properties whether it is a dict or a Pydantic Model
    cat = action.get("category", "") if isinstance(action, dict) else getattr(action, "category", "")
    pri = action.get("priority", "") if isinstance(action, dict) else getattr(action, "priority", "")
    resp = action.get("response_snippet", "") if isinstance(action, dict) else getattr(action, "response_snippet", "")

    return f"""You are an strict Customer Support QA Agent Grader.
Evaluate the AI's triage results for the following ticket:

--- Ticket ---
Subject: {ticket.get('subject', '')}
Body: {ticket.get('body', '')}
Tier: {ticket.get('customer_tier', '')}

--- Expected Routing ---
Category: {expected.get('category', '')}
Priority: {expected.get('priority', '')}
Key Topics: {', '.join(expected.get('response_keywords', []))}

--- AI Agent's Action ---
Category: {cat}
Priority: {pri}
Response Snippet: {resp}

Evaluate the action's accuracy. 
- Award 0.4 if the category perfectly matches.
- Award 0.3 if the priority perfectly matches.
- Award up to 0.3 based on the quality, semantic relevance, and inclusion of key topics in the response snippet.
Provide partial progress scores if the response is somewhat relevant but misses key specifics.

Output ONLY a single float value between 0.0 and 1.0 (e.g. "0.85"). Do not output any explanation.
"""

def evaluate_with_llm(ticket, expected, action) -> float:
    try:
        prompt = create_grader_prompt(ticket, expected, action)
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        score_str = response.choices[0].message.content.strip()
        # Parse the float
        import re
        match = re.search(r"0\.\d+|1\.0", score_str)
        if match:
            return float(match.group())
        return float(score_str)
    except Exception as e:
        print(f"LLM Grader Exception: {e}")
        # Deterministic fallback mechanism
        cat = action.get("category", "") if isinstance(action, dict) else getattr(action, "category", "")
        if cat.lower() == expected["category"].lower(): return 0.5
        return 0.2

def grade(*args, **kwargs) -> float:
    """Agent Grader for Task 001."""
    # Try different fallback injection points based on validator structure
    action = kwargs.get("action", args[0] if args else {})
    expected = kwargs.get("expected", {
            "category": "billing",
            "priority": "medium",
            "response_keywords": ["invoice", "charge", "billing"]
    })
    ticket = kwargs.get("ticket", {
            "subject": "Billing charge question",
            "body": "Hi, I was charged $49.99 on my last invoice but I expected $29.99."
    })
    return evaluate_with_llm(ticket, expected, action)
