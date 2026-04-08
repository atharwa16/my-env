"""Baseline inference script for the Support Ticket Triage environment.

Runs locally (no server needed) and prints logs in the required
[START] / [STEP] / [END] format.
"""

from models import Action
from support_env import SupportTriageEnv


TASK_NAME = "support-ticket-triage"
ENV_NAME = "support-triage-env"
MODEL_NAME = "baseline-rule-agent"


def make_baseline_action(observation) -> Action:
    """A simple rule-based baseline agent.

    Examines the ticket body for keywords to decide category and priority.
    """
    body = observation.body.lower()
    subject = observation.subject.lower()
    text = body + " " + subject

    # Category heuristics
    if "billing" in text or "charge" in text or "invoice" in text:
        category = "billing"
    elif "password" in text or "login" in text or "access" in text or "dashboard" in text:
        category = "technical"
    elif "account" in text or "downgrade" in text or "upgrade" in text:
        category = "account"
    else:
        category = "general"

    # Priority heuristics
    if "urgent" in text or "block" in text or "tomorrow" in text:
        priority = "urgent"
    elif observation.customer_tier == "enterprise":
        priority = "high"
    elif "charge" in text or "billing" in text:
        priority = "medium"
    else:
        priority = "low"

    # Response snippet: echo back some keywords from the ticket
    keywords = []
    for kw in ["invoice", "charge", "billing", "password", "access",
                "login", "downgrade", "account"]:
        if kw in text:
            keywords.append(kw)

    snippet = f"We understand your concern regarding {', '.join(keywords[:3]) if keywords else 'your issue'}. We will look into it."

    return Action(category=category, priority=priority, response_snippet=snippet)


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
            action = make_baseline_action(obs)
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

        print(f"[END] success={success_str} steps={step_num} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    main()
