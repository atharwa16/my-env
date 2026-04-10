import os
import difflib

def _get_priority_val(p_str: str) -> int:
    mapping = {"low": 1, "medium": 2, "high": 3, "urgent": 4, "critical": 5}
    return mapping.get(str(p_str).lower().strip(), 0)

def _get_keyword_score(resp_snippet: str, expected_keywords: list) -> float:
    """Programmatically score the response based on keyword presence and similarity."""
    if not expected_keywords:
        return 1.0
    
    resp_lower = str(resp_snippet).lower()
    found_score = 0.0
    
    # Clean response for word splitting
    clean_resp = resp_lower.replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ')
    words = clean_resp.split()
    
    for kw in expected_keywords:
        kw_lower = kw.lower().strip()
        # Direct substring match (handles phrases)
        if kw_lower in resp_lower:
            found_score += 1.0
        else:
            # Fuzzy match individual words for partial credit
            best_word_match = 0.0
            for word in words:
                ratio = difflib.SequenceMatcher(None, kw_lower, word).ratio()
                if ratio > best_word_match:
                    best_word_match = ratio
            
            if best_word_match > 0.8: # High threshold for typos/variants
                found_score += best_word_match * 0.5 # 50% credit for near miss (0.5 weight)
                
    return min(1.0, found_score / len(expected_keywords))

def grade(*args, **kwargs) -> float:
    """Universal 100% Programmatic Agent Grader for all tasks."""
    action = kwargs.get("action", args[0] if args else {})
    expected = kwargs.get("expected", {})
    # Note: 'ticket' info is no longer needed for a programmatic keyword grader
    
    # Helper to extract values from dict or object
    def get_val(obj, key, default=""):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # --- 1. Category Match (40%) ---
    cat_exp = str(expected.get("category", "")).strip().lower()
    cat_act = str(get_val(action, "category")).strip().lower()
    
    if cat_exp == cat_act:
        cat_score = 1.0
    elif {cat_exp, cat_act} == {"account", "billing"}:
        cat_score = 0.8  # Objective: Account/Billing crossover logic
    else:
        cat_score = difflib.SequenceMatcher(None, cat_exp, cat_act).ratio()
    
    # --- 2. Priority Match (30%) ---
    pri_exp = str(expected.get("priority", "")).strip().lower()
    pri_act = str(get_val(action, "priority")).strip().lower()
    
    v_exp, v_act = _get_priority_val(pri_exp), _get_priority_val(pri_act)
    if v_exp and v_act:
        # Objective: Distance calculation based on 5 levels (0.25 per level)
        pri_score = max(0.0, 1.0 - (abs(v_exp - v_act) * 0.25))
    else:
        pri_score = difflib.SequenceMatcher(None, pri_exp, pri_act).ratio()

    # --- 3. Keyword Match (30%) ---
    resp_snippet = get_val(action, "response_snippet")
    keywords = expected.get("response_keywords", [])
    kw_score = _get_keyword_score(resp_snippet, keywords)
        
    final_score = (cat_score * 0.4) + (pri_score * 0.3) + (kw_score * 0.3)
    return float(min(1.0, max(0.0, round(final_score, 3))))
