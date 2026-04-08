"""Pydantic models for the Support Ticket Triage environment."""

from pydantic import BaseModel, Field
from typing import Optional


class Observation(BaseModel):
    """What the agent sees: a support ticket."""
    ticket_id: str = Field(..., description="Unique ticket identifier")
    subject: str = Field(..., description="Ticket subject line")
    body: str = Field(..., description="Ticket body text")
    customer_tier: str = Field(..., description="Customer tier: free, pro, enterprise")
    task_difficulty: str = Field(..., description="Task difficulty: easy, medium, hard")


class Action(BaseModel):
    """What the agent does: triage the ticket."""
    category: str = Field(..., description="Predicted category: billing, technical, account, general")
    priority: str = Field(..., description="Predicted priority: low, medium, high, urgent")
    response_snippet: str = Field(..., description="Short response snippet for the customer")


class StepResult(BaseModel):
    """Result returned after the agent takes an action."""
    observation: Optional[Observation] = Field(None, description="Next observation (None if done)")
    reward: float = Field(..., ge=0.0, le=1.0, description="Reward between 0.0 and 1.0")
    done: bool = Field(..., description="Whether the episode is finished")
    info: dict = Field(default_factory=dict, description="Extra info about scoring breakdown")


class EnvState(BaseModel):
    """Current state of the environment."""
    current_task_index: int = Field(..., description="Index of the current task")
    total_tasks: int = Field(..., description="Total number of tasks")
    done: bool = Field(..., description="Whether all tasks are completed")
    cumulative_reward: float = Field(..., description="Total reward accumulated so far")
    current_observation: Optional[Observation] = Field(None, description="Current observation")
