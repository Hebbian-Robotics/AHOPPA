"""Human-readable instructions for the scene's task, without evaluator coordinates."""

from humanoid_agent.shutdown_task import ShutdownTask, shutdown_instruction


def scene_task_instruction(requested: str | None, shutdown: ShutdownTask) -> str:
    """An explicit instruction overrides the scene's shutdown instruction."""
    return shutdown_instruction(shutdown) if requested is None else requested
