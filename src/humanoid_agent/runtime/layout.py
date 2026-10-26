"""Locate per-episode evidence independently of the number of simulated robots."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpisodeLayout:
    root: Path
    num_environments: int
    single_episode: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.num_environments <= 16:
            raise ValueError("An episode owner requires 1–16 environments")
        if self.single_episode and self.num_environments != 1:
            raise ValueError("A single episode directory requires exactly one environment")

    @property
    def physics_directory(self) -> Path:
        return self.root / "shared-physics"

    @property
    def episode_directories(self) -> tuple[Path, ...]:
        if self.single_episode:
            return (self.root,)
        return tuple(
            self.root / "episodes" / f"env_{index}" for index in range(self.num_environments)
        )

    def camera_directory(self, environment_index: int) -> Path:
        if not 0 <= environment_index < self.num_environments:
            raise ValueError("Camera environment index is outside the episode owner")
        directory = self.physics_directory / "camera-output"
        return directory if self.num_environments == 1 else directory / f"env_{environment_index}"
