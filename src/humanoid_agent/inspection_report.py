"""Optional report field retained in every model decision.

The earlier signpost-inspection task added ``inspection_report`` to the shared
planning decision. The drawer shutdown round never scores it, but the field
stays in the schema so model requests remain identical to the recorded ones.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from humanoid_agent.validated_types import ImmutableSequence, ValidatedModel

StationIdentifier = Literal["A", "B", "C"]
IndicatorValue = Literal["B", "C", "NORMAL", "MAINTENANCE"]


class StationReport(ValidatedModel):
    station: StationIdentifier
    indicator: IndicatorValue


class InspectionReport(ValidatedModel):
    observations: ImmutableSequence[StationReport] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def require_unique_stations(self) -> InspectionReport:
        identifiers = [observation.station for observation in self.observations]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Report each station at most once")
        return self
