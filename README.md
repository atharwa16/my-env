---
title: Support Ticket Triage
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# Support Ticket Triage Environment

An OpenEnv-compliant reinforcement learning environment for automated customer support ticket triage, powered by dynamic LLM-as-a-judge agent evaluators.

## Motivation & Real-World Utility
Customer support centers handle massive volumes of incoming tickets that require immediate sorting, prioritization, and initial acknowledgement. Delays or miscategorization can severely impact enterprise SLAs and customer satisfaction. This environment trains and evaluates AI agents on their ability to accurately parse complex support requests, route them to the correct technical departments, assign appropriate urgencies based on context and customer tiers, and draft context-aware response snippets.

## Environment Overview
The environment natively conforms to standard Reinforcement Learning paradigms isolated around a standard HTTP-based loop:
- **`reset()`**: Initializes the environment to a specific task index (ticket observation). Resets cumulative rewards and flags the episode as active.
- **`step(action)`**: Submits the agent's Pydantic `Action` payload to the true LLM agent grader. Calculates the reward based strictly on the evaluation criteria, and immediately sets `done=True` as each ticket is treated as an isolated one-step episode.
- **`done`**: Flags the termination of the current ticket episode, requiring a subsequent `reset()` structural call to progress to the next ticket.

## Action Space
The Action Space defines the payload the agent must submit to successfully triage the ticket. It expects an `Action` Pydantic model with the following strictly typed string fields:
- `category` (string): The predicted department routing. Must be one of `billing`, `technical`, `account`, `general`.
- `priority` (string): The predicted urgency. Must be one of `low`, `medium`, `high`, `urgent`, `critical`.
- `response_snippet` (string): A short generated response string for the customer acknowledging their issue.

## Observation Space
The Observation Space represents the incoming support ticket that the agent must analyze. It returns an `Observation` Pydantic model containing:
- `ticket_id` (string): A unique ticket identifier (e.g., "TKT-001").
- `subject` (string): The subject line of the customer's email.
- `body` (string): The full multiline complaint or query body text.
- `customer_tier` (string): The customer's subscription level (e.g., `free`, `pro`, `enterprise`), which heavily influences priority.
- `task_difficulty` (string): Evaluated difficulty level (`easy`, `medium`, `hard`).

## Reward Function
The environment evaluates the agent's triage using a robust partial progress signal scaled tightly between `0.0` and `1.0`. The actual scoring is explicitly evaluated natively by an LLM-Agent-as-a-judge (`server/grader.py`) configured to measure:
1. **0.4 Points (Category Check)**: Awarded for a category match. Includes semantic awareness; e.g., predicting `billing` for an `account` issue (or vice versa) still receives **0.8 reward weight (0.32 pts)** due to their high correlation in support workflows.
2. **0.3 Points (Priority Check)**: Awarded based on proximity within the 5-level hierarchy: `critical > urgent > high > medium > low`.
   - Exact match: **1.0 (0.30 pts)**
   - 1 level distance (e.g., High vs Urgent): **0.75 (0.225 pts)**
   - 2 levels distance: **0.50 (0.15 pts)**
   - etc.
3. **Up to 0.3 Points (Semantic Quality)**: Evaluated by an LLM-Agent-as-a-judge to ensure the `response_snippet` is contextually relevant, polite, and addresses key ticket themes.

## Tasks
The `openenv.yaml` manifest defines a progression of 3 increasingly difficult deterministic tasks:
1. **TKT-001 (Easy)**: A highly specific billing charge question for a "pro" customer asking about a $49.99 charge.
2. **TKT-002 (Medium)**: A password reset issue for an "enterprise" customer. The high customer tier forces the priority to be elevated to 'high' despite password resets generally being common.
3. **TKT-003 (Hard)**: Ambigous account downgrade complaints resulting in access loss and unexpected billing variables. Requires deep, multifaceted text comprehension to correctly assess as an "urgent" "account" issue.

## Agent Grader Logic
To satisfy rigorous Phase 2 LLM-evaluation tests, the codebase utilizes a unified Agent Grader (`server/grader.py`). Instead of basic regex grading, the `step()` function injects the system prompt into a live Gemini OpenAI-compatible client. The client is explicitly instructed to act as a strict Customer Support QA. It absorbs the ticket context alongside the agent's action and enforces the partial progress scoring matrix dynamically. 

If the API fails to connect (e.g., due to local execution missing proxy credentials), the script silently defaults to a strict deterministic fallback calculating fixed categorical string matching returning either `0.5` or `0.2` to natively prevent runtime crashes.

## Example Interaction
**Action (Request `POST /step`)**
```json
{
  "category": "technical",
  "priority": "high",
  "response_snippet": "We apologize for the login trouble. Because you are an enterprise customer, we will escalate this password reset to our technical team immediately."
}
```

**StepResult (Response from `POST /step`)**
```json
{
  "observation": null,
  "reward": 0.85,
  "done": true,
  "info": {
    "llm_eval": 0.85,
    "category_match": true
  }
}
```

## Setup Instructions

**Install Dependencies:**
```bash
# Python 3.10+ required
pip install -r requirements.txt
```

**Run Locally:**
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Docker Build & Run:**
```bash
docker build -t support-triage-env .
docker run -p 7860:7860 support-triage-env
```

## Deployment
The application is pre-configured for a seamless multi-mode deployment, prioritizing accessibility via Hugging Face Spaces.
- **Hugging Face Space**: Deployable natively utilizing the included `Dockerfile` pointing directly to port `7860`.
- **API Endpoints**: The underlying FastAPI runtime exposes `/reset` (to establish isolated RL tasks), `/step` (to accept the Pydantic action JSON and formulate the gradient reward), and `/state`.

## Inference Script
The project root contains `inference.py`, a robust baseline script utilizing the `OpenAI` python client to directly interface with an LLM. It isolates the environment explicitly into 3 distinct iterations. It expects the following OpenEnv-compliant environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (or `API_KEY` / `GEMINI_API_KEY` specific proxies)

## Logging Format (STRICT)
The inference script operates via strict standard output formatting required by automated validators to cleanly trace testing lifecycles. It completely isolates logs chronologically per episode utilizing:
```
[START] task=tkt_001 env=support-triage-env model=gemini-2.0-flash
[STEP] step=1 action=category=billing|priority=medium|response=... reward=0.85 done=true error=null
[END] success=true steps=1 score=0.85 rewards=0.85
```

## Reproducibility
The environment tasks internally are completely deterministic, loaded explicitly from dictionary-bound JSON logic (`server/tasks.py`). Because evaluations isolate index targeting structurally (`env.reset(task_index=i)`), identical actions sequentially tested inside the loop consistently yield precisely mapped outputs and deterministic LLM semantic grading bounds.

## Validation Status
- Hugging Face Space responds correctly
- Docker builds successfully
- `openenv validate` passes locally
- Hackathon Phase 2 evaluation checks **PASSED**
