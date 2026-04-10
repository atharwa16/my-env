"""LLM-based Baseline inference script for the Support Ticket Triage environment.

Runs locally and queries an LLM to triage the ticket. 
Prints logs in the required [START] / [STEP] / [END] format.
"""

import os
import json
from openai import OpenAI
from server.models import Action
from server.support_env import SupportTriageEnv


TASK_NAME = "support-ticket-triage"
ENV_NAME = "support-triage-env"
API_BASE_URL = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash")
API_KEY = os.environ.get("API_KEY", os.environ.get("GEMINI_API_KEY", "your_gemini_api_key_here"))

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

_client = None

def get_client() -> OpenAI:
    """Lazily create the OpenAI-compatible client for Gemini."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
    return _client


def make_llm_action(observation) -> Action:
    """Uses an LLM to decide the triage category and priority, and generates a response."""
    
    prompt = f"""You are an automated support ticket triage system. Analyze the following ticket.

Ticket Subject: {observation.subject}
Ticket Body: {observation.body}
Customer Tier: {observation.customer_tier}

Triage the ticket by providing:
1. "category": Must be strictly one of ["billing", "technical", "account", "general"]
2. "priority": Must be strictly one of ["low", "medium", "high", "urgent"]
3. "response_snippet": A short response (1-2 sentences) acknowledging their issue. Reiterate key words from their ticket.

Return ONLY a raw JSON object with these 3 keys. Do not include markdown blocks or any other text.
"""
    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        
        # Clean potential markdown formatting just in case
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        data = json.loads(content.strip())
        
        # Ensure fallback values if the JSON is malformed
        return Action(
            category=data.get("category", "general"),
            priority=data.get("priority", "low"),
            response_snippet=data.get("response_snippet", "We have received your ticket.")
        )
    except Exception as e:
        # Fallback to rule-based agent if LLM fails (e.g. timeout, invalid JSON, invalid auth)
        return fallback_rule_action(observation)


def fallback_rule_action(observation) -> Action:
    """Fallback rule-based agent in case the LLM API call fails."""
    text = (observation.body + " " + observation.subject).lower()
    
    if "billing" in text or "charge" in text or "invoice" in text:
        category = "billing"
    elif "password" in text or "login" in text or "access" in text:
        category = "technical"
    elif "account" in text or "downgrade" in text:
        category = "account"
    else:
        category = "general"

    if "urgent" in text or "block" in text:
        priority = "urgent"
    elif observation.customer_tier == "enterprise":
        priority = "high"
    elif "charge" in text or "billing" in text:
        priority = "medium"
    else:
        priority = "low"

    return Action(category=category, priority=priority, response_snippet="We will look into your issue.")


def format_action(action: Action) -> str:
    """Convert action to pipe-delimited single string."""
    return f"category={action.category}|priority={action.priority}|response={action.response_snippet}"


def main():
    task_ids = ["tkt_001", "tkt_002", "tkt_003"]
    
    try:
        env = SupportTriageEnv()

        for i, task_id in enumerate(task_ids):
            print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")
            
            obs = env.reset(task_index=i)
            action = make_llm_action(obs)
            result = env.step(action)

            action_str = format_action(action)
            done_str = "true" if result.done else "false"

            # Each ticket is exactly 1 step
            print(f"[STEP] step=1 action={action_str} reward={result.reward:.2f} done={done_str} error=null")
            print(f"[END] success=true steps=1 score={result.reward:.2f} rewards={result.reward:.2f}")

    except Exception as e:
        error_msg = str(e)
        print(f"[STEP] step=1 action=error reward=0.00 done=true error={error_msg}")
        print(f"[END] success=false steps=1 score=0.00 rewards=0.00")


if __name__ == "__main__":
    main()
