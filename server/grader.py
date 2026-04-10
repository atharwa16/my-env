import os
import difflib
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
API_KEY = os.getenv("API_KEY", os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here"))

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _client

def _get_priority_val(p_str: str) -> int:
    mapping = {"low": 1, "medium": 2, "high": 3, "urgent": 4, "critical": 5}
    return mapping.get(str(p_str).lower().strip(), 0)

def create_grader_prompt(ticket, expected, action) -> str:
    resp = action.get("response_snippet", "") if isinstance(action, dict) else getattr(action, "response_snippet", "")

    return f"""You are a Customer Support QA Agent Grader.
Evaluate ONLY the semantic quality and relevance of the AI's response snippet.

--- Ticket ---
Subject: {ticket.get('subject', '')}
Body: {ticket.get('body', '')}
Tier: {ticket.get('customer_tier', '')}

--- Expected Key Topics ---
{', '.join(expected.get('response_keywords', []))}

--- AI Agent's Response Snippet ---
{resp}

Evaluate the snippet out of 1.0. 
Award proportional credit (e.g., 0.3, 0.6) based on how well it addresses the ticket and covers expected key topics.
Output ONLY a single float value between 0.0 and 1.0 (e.g. "0.85"). Do not output any explanation.
"""

def grade(*args, **kwargs) -> float:
    """Universal Agent Grader for all tasks."""
    action = kwargs.get("action", args[0] if args else {})
    expected = kwargs.get("expected", {})
    ticket = kwargs.get("ticket", {})
    
    # --- Programmatic Smooth Scoring ---
    cat_exp = str(expected.get("category", "")).strip().lower()
    cat_act = str(action.get("category", "") if isinstance(action, dict) else getattr(action, "category", "")).strip().lower()
    
    # Semantic Category Check
    if cat_exp == cat_act:
        cat_score = 1.0
    elif {cat_exp, cat_act} == {"account", "billing"}:
        cat_score = 0.8  # Consider account and billing closely related
    else:
        cat_score = difflib.SequenceMatcher(None, cat_exp, cat_act).ratio()
    
    pri_exp = str(expected.get("priority", "")).strip().lower()
    pri_act = str(action.get("priority", "") if isinstance(action, dict) else getattr(action, "priority", "")).strip().lower()
    
    v_exp, v_act = _get_priority_val(pri_exp), _get_priority_val(pri_act)
    if v_exp and v_act:
        # Distance calculation based on 5 levels (max distance 4)
        pri_score = max(0.0, 1.0 - (abs(v_exp - v_act) * 0.25))
    else:
        pri_score = difflib.SequenceMatcher(None, pri_exp, pri_act).ratio()

    # --- LLM Response Quality Evaluation ---
    llm_resp_score = 0.0
    try:
        prompt = create_grader_prompt(ticket, expected, action)
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        score_str = response.choices[0].message.content.strip()
        import re
        match = re.search(r"0\.\d+|1\.0", score_str)
        if match:
            llm_resp_score = float(match.group())
        else:
            llm_resp_score = float(score_str)
    except Exception as e:
        print(f"LLM Grader Exception: {e}")
        llm_resp_score = 0.2
        
    final_score = (cat_score * 0.4) + (pri_score * 0.3) + (llm_resp_score * 0.3)
    return float(min(1.0, max(0.0, round(final_score, 3))))
