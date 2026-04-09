---
title: Support Ticket Triage
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# Customer Support Ticket Triage — RL Environment

An OpenEnv-compliant reinforcement learning environment where an agent learns to triage customer support tickets by assigning a category, priority level, and response.

## Motivation

Customer support is one of the largest operational costs for SaaS platforms, e-commerce companies, and service providers. As ticket volume scales, manual triage becomes a bottleneck — misrouted tickets increase resolution time, and incorrect prioritization leads to SLA breaches.

This environment frames ticket triage as a reinforcement learning task: an agent receives a raw support ticket and must classify it accurately and respond appropriately. The goal is to build agents that can handle the ambiguity, noise, and urgency signals found in real customer messages.

## Environment Description

The environment simulates a support ticket queue. On each episode, the agent receives a ticket (observation) and must produce a triage decision (action). The environment scores the action against a ground-truth label using a deterministic grader and returns a reward.

**Interaction loop:**

```
observation = env.reset()       # Get the first ticket
result = env.step(action)       # Submit triage, receive reward
# result contains: observation, reward, done, info
```

The environment exposes three core methods:

| Method | Description |
|--------|-------------|
| `reset()` | Reset to the first task, return initial observation |
| `step(action)` | Submit a triage action, receive reward and next observation |
| `state()` | Inspect current environment state |

## Observation Space

Each observation is a support ticket with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | string | Unique ticket identifier |
| `subject` | string | Ticket subject line |
| `body` | string | Full ticket text from the customer |
| `customer_tier` | string | Customer plan: `free`, `pro`, `enterprise` |
| `task_difficulty` | string | Difficulty level: `easy`, `medium`, `hard` |

The agent must read the ticket text and infer the correct triage from context, tone, and keywords.

## Action Space

The agent produces a triage decision with three components:

| Field | Type | Valid Values |
|-------|------|-------------|
| `category` | string | `billing`, `technical`, `account`, `general` |
| `priority` | string | `low`, `medium`, `high`, `urgent` |
| `response_snippet` | string | Free-text response to the customer |

## Reward Function

Reward is computed deterministically using three components. Total reward is always in the range **[0.0, 1.0]**.

| Component | Weight | Criteria |
|-----------|--------|----------|
| Category match | **0.4** | Exact match with expected category |
| Priority match | **0.3** | Exact match with expected priority |
| Response keyword match | **0.3** | Fraction of expected keywords found in response snippet |

**Example:** If the agent matches the category correctly (0.4), misses the priority (0.0), and hits 2 of 3 keywords (0.2), the total reward is **0.60**.

## Tasks

The environment includes 3 tasks with increasing difficulty:

### Task 1 — Easy

> *"I was charged $49.99 on my last invoice but I expected $29.99."*

A clear billing question. Category, priority, and keywords are straightforward.

### Task 2 — Medium

> *"I reset my password yesterday and now I can't log into the dashboard. I'm an enterprise customer and this is blocking my team."*

A technical issue with an urgency signal. The enterprise tier and blocking language add ambiguity around priority.

### Task 3 — Hard

> *"I downgraded from Enterprise to Pro last week. Since then, several team members lost access... and I'm still being charged the Enterprise rate."*

A multi-issue ticket spanning account access and billing. The agent must identify the dominant category and parse urgency from context.

## Episode Design

Each episode processes **one ticket at a time**:

1. The agent receives a ticket via `reset()` or the previous `step()` result.
2. The agent submits one action (triage decision).
3. The environment returns a reward and the next ticket (or `done=true`).

After all 3 tasks are completed, the episode ends. Call `reset()` to start again.

## Setup Instructions

### Docker (recommended)

```bash
docker build -t support-triage-env .
docker run -p 7860:7860 support-triage-env
```

The FastAPI server will be available at `http://localhost:7860`.

### Test endpoints

```bash
# Health check
curl http://localhost:7860/health

# Reset and get first ticket
curl -X POST http://localhost:7860/reset

# Submit a triage action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"category":"billing","priority":"medium","response_snippet":"We will review your invoice and charge."}'
```

### Local (without Docker)

Note: Python 3.12 (or at least 3.10+) is required by `openenv-core`.

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Usage

First, specify your LLM environment variables (the setup comes pre-configured for the Gemini API via its OpenAI-compatible endpoint):

```bash
export GEMINI_API_KEY="your_gemini_api_key_here" # Your actual Gemini API key
export MODEL_NAME="gemini-2.0-flash"
# Optional overrides:
# export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

Run the baseline inference script:

```bash
python inference.py
```

This runs an LLM-based agent via the OpenAI python client that evaluates all 3 tickets locally and prints results in the exact evaluation format:

```
[START] task=support-ticket-triage env=support-triage-env model=gemini-2.0-flash
[STEP] step=1 action=category=billing|priority=medium|response=... reward=0.70 done=false error=null
[STEP] step=2 action=category=technical|priority=urgent|response=... reward=0.40 done=false error=null
[STEP] step=3 action=category=billing|priority=urgent|response=... reward=0.30 done=true error=null
[END] success=true steps=3 score=0.47 rewards=0.70,0.40,0.30
```

## Baseline Results

| Task | Difficulty | Reward | Notes |
|------|-----------|--------|-------|
| TKT-001 | Easy | 1.00 | Perfect: category, priority, and all keywords matched |
| TKT-002 | Medium | 0.60 | Example fallback: Category correct, priority wrong |
| TKT-003 | Hard | 0.50 | Example fallback: Category wrong, priority correct |

**Average score: 0.70 (Fallback limits), up to 1.00 (Perfect LLM)**

The baseline uses an LLM. It includes a rule-based fallback just in case the LLM API call fails, preventing your evaluation from completely crashing to 0.00.

## Future Scope

- **Semantic grading** — Replace keyword matching with embedding similarity for more robust response evaluation.
- **Real-world datasets** — Integrate tickets from public datasets (e.g., customer support on Twitter) for realistic noise and scale.
- **Multi-turn episodes** — Extend to conversational triage where the agent can ask follow-up questions.
- **Dynamic difficulty** — Generate tickets procedurally with varying ambiguity levels.

## Project Structure

```
├── server/
│   ├── app.py              # FastAPI server (/reset, /step, /state, /health)
│   ├── support_env.py      # Environment class (reset, step, state)
│   ├── models.py           # Pydantic models (Observation, Action, StepResult, EnvState)
│   └── tasks.py            # 3 deterministic tasks (easy, medium, hard)
├── inference.py        # Baseline agent with [START]/[STEP]/[END] logging
├── openenv.yaml        # OpenEnv manifest with specific task graders
├── pyproject.toml      # Project configuration and specs
├── Dockerfile          # Container definition (port 7860)
├── requirements.txt    # Python dependencies
└── README.md
```
