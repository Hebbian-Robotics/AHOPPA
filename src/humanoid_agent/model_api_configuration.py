"""CLI configuration shared by physics and model processes without importing SDKs."""

from enum import StrEnum


class ApiServiceTier(StrEnum):
    STANDARD = "standard"
    FLEX = "flex"


class ReasoningEffort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ApiRetryPolicy(StrEnum):
    NONE = "none"
    BOUNDED = "bounded-api-retry-v1"


RECORDED_API_MAXIMUM_ATTEMPTS = 2
RECORDED_API_RETRY_WAIT_SECONDS = 1.0
