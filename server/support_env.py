"""Support Ticket Triage environment logic."""

from typing import Optional

from .models import Observation, Action, StepResult, EnvState
from .tasks import TASKS


class SupportTriageEnv:
    """OpenEnv-compliant RL environment for customer support ticket triage."""

    def __init__(self):
        self._tasks = TASKS
        self._current_index = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._current_observation: Optional[Observation] = None

    def reset(self) -> Observation:
        """Reset the environment to the first task and return the first observation."""
        self._current_index = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._current_observation = Observation(**self._tasks[0]["ticket"])
        return self._current_observation

    def step(self, action: Action) -> StepResult:
        """Take an action (triage) on the current ticket and return the result."""
        if self._done:
            return StepResult(
                observation=None,
                reward=0.0,
                done=True,
                info={"error": "Environment is done. Call reset()."},
            )

        task = self._tasks[self._current_index]
        expected = task["expected"]

        # --- Deterministic grading ---
        reward = 0.0
        breakdown = {}

        # Category match: 0.4 points
        if action.category.lower() == expected["category"]:
            reward += 0.4
            breakdown["category"] = 0.4
        else:
            breakdown["category"] = 0.0

        # Priority match: 0.3 points
        if action.priority.lower() == expected["priority"]:
            reward += 0.3
            breakdown["priority"] = 0.3
        else:
            breakdown["priority"] = 0.0

        # Response keyword match: 0.3 points (partial — per keyword)
        expected_keywords = expected["response_keywords"]
        snippet_lower = action.response_snippet.lower()
        matched = sum(1 for kw in expected_keywords if kw in snippet_lower)
        keyword_score = round(0.3 * (matched / len(expected_keywords)), 4)
        reward += keyword_score
        breakdown["response_keywords"] = keyword_score
        breakdown["keywords_matched"] = f"{matched}/{len(expected_keywords)}"

        reward = round(reward, 4)
        self._cumulative_reward += reward

        # Advance to next task or finish
        self._current_index += 1
        if self._current_index >= len(self._tasks):
            self._done = True
            self._current_observation = None
            return StepResult(
                observation=None,
                reward=reward,
                done=True,
                info=breakdown,
            )

        self._current_observation = Observation(**self._tasks[self._current_index]["ticket"])
        return StepResult(
            observation=self._current_observation,
            reward=reward,
            done=False,
            info=breakdown,
        )

    def state(self) -> EnvState:
        """Return the current state of the environment."""
        return EnvState(
            current_task_index=self._current_index,
            total_tasks=len(self._tasks),
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            current_observation=self._current_observation,
        )
