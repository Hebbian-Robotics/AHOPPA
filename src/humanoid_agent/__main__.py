"""Run the installed humanoid-agent command."""

from humanoid_agent.cli import main
from humanoid_agent.logging_config import configure_logging

if __name__ == "__main__":
    configure_logging()
    main()
