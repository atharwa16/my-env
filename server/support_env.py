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

    def reset(self, task_index: int = 0) -> Observation:
        """Reset the environment to a specific task safely."""
        self._current_index = task_index
        self._done = False
        self._cumulative_reward = 0.0
        self._current_observation = Observation(**self._tasks[self._current_index]["ticket"])
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

        # --- Agent Grader Integration ---
        from .grader import grade as universal_agent_grader

        # Execute True LLM Agent Evaluation Output
        reward = universal_agent_grader(action=action, expected=expected, ticket=task["ticket"])
        reward = round(reward, 4)

        # Build basic summary info
        breakdown = {"llm_eval": reward, "category_match": action.category.lower() == expected["category"]}
        
        self._cumulative_reward += reward

        # Mark the episode as finished because each task is exactly 1 step
        self._done = True
        self._current_observation = None
        return StepResult(
            observation=None,
            reward=reward,
            done=True,
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
