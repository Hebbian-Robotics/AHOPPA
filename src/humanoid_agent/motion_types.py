"""Shared numerical motion schemas and native provider usage accounting."""

from pydantic import BaseModel, ConfigDict, Field

from humanoid_agent.validated_types import ImmutableSequence, ValidatedModel, Vector3

Vector = Vector3


class TrackingPose(ValidatedModel):
    left_wrist: Vector
    right_wrist: Vector
    head: Vector


class HybridChunk(ValidatedModel):
    walking: bool
    speed_m_per_s: float = Field(ge=0.1, le=0.8)
    movement_direction: Vector
    facing_yaw_radians: float = Field(ge=-3.141593, le=3.141593)
    poses: ImmutableSequence[TrackingPose] = Field(min_length=4, max_length=4)


class MotionUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
