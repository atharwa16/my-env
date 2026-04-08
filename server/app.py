"""FastAPI server exposing the Support Ticket Triage environment."""

from fastapi import FastAPI, HTTPException
from .models import Action, StepResult, EnvState, Observation
from .support_env import SupportTriageEnv

app = FastAPI(
    title="Support Ticket Triage Environment",
    description="OpenEnv-compliant RL environment for customer support ticket triage.",
    version="1.0.0",
)

env = SupportTriageEnv()


@app.get("/")
def read_root():
    """Root endpoint for Hugging Face Spaces."""
    return {"message": "Support Ticket Triage Environment is running. Access /docs for the API reference."}


@app.post("/reset", response_model=Observation)
def reset():
    """Reset the environment and return the first observation."""
    return env.reset()


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """Submit a triage action and receive the step result."""
    return env.step(action)


@app.get("/state", response_model=EnvState)
def state():
    """Get the current environment state."""
    return env.state()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

