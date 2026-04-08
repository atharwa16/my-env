"""LLM-based Baseline inference script for the Support Ticket Triage environment.

Runs locally and queries an LLM to triage the ticket. 
Prints logs in the required [START] / [STEP] / [END] format.
"""

import os
import json
from openai import OpenAI
from models import Action
from support_env import SupportTriageEnv


TASK_NAME = "support-ticket-triage"
ENV_NAME = "support-triage-env"
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


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
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
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
    rewards = []
    step_num = 0
    success = True
    error_msg = "null"

    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}")

    try:
        env = SupportTriageEnv()
        obs = env.reset()

        while True:
            action = make_llm_action(obs)
            result = env.step(action)
            step_num += 1

            action_str = format_action(action)
            done_str = "true" if result.done else "false"

            print(f"[STEP] step={step_num} action={action_str} reward={result.reward:.2f} done={done_str} error=null")

            rewards.append(result.reward)

            if result.done:
                break

            obs = result.observation

    except Exception as e:
        success = False
        error_msg = str(e)
        step_num += 1
        print(f"[STEP] step={step_num} action=error reward=0.00 done=true error={error_msg}")
        rewards.append(0.0)

    finally:
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success_str = "true" if success else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        # Truncate and sanitize environment error messages on completion if needed
        print(f"[END] success={success_str} steps={step_num} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    main()
