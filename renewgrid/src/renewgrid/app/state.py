"""App state types for the Resilience Lab UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date


@dataclass(frozen=True)
class RunConfig:
    """Serializable run configuration for guided and advanced workflows."""

    region: str
    start_date: date
    end_date: date
    timeframe_preset: str
    use_rare: bool
    rare_path: str | None
    model_choice: str
    horizon_days: int

    def to_dict(self) -> dict[str, object]:
        """Convert config into JSON-safe dictionary."""
        payload = asdict(self)
        payload["start_date"] = self.start_date.isoformat()
        payload["end_date"] = self.end_date.isoformat()
        return payload
