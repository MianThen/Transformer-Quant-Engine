from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("registry timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    family_id: str
    trial_id: str
    method_version: str
    portfolio_policy_id: str
    policy_config_hash: int
    expected_direction: str
    primary_metric: str
    validation_windows: tuple[str, ...]
    p_value_method: str
    created_before_test_at: str

    def validate(self, test_data_available_at: str | None = None) -> None:
        for value in (self.hypothesis_id, self.family_id, self.trial_id):
            if not _ID_PATTERN.fullmatch(value):
                raise ValueError("hypothesis, family, and trial IDs must be stable ASCII IDs")
        if not self.method_version or not self.portfolio_policy_id or not self.primary_metric:
            raise ValueError("hypothesis method, policy, and metric are required")
        if self.policy_config_hash <= 0:
            raise ValueError("policy_config_hash must be non-zero")
        if self.expected_direction not in {"POSITIVE", "NEGATIVE"}:
            raise ValueError("expected_direction must be POSITIVE or NEGATIVE")
        if not self.validation_windows or len(set(self.validation_windows)) != len(
            self.validation_windows
        ):
            raise ValueError("validation_windows must be non-empty and unique")
        if not self.p_value_method:
            raise ValueError("p_value_method is required")
        created = _timestamp(self.created_before_test_at)
        if test_data_available_at is not None and created >= _timestamp(test_data_available_at):
            raise ValueError("hypothesis must be registered before test data is available")


@dataclass(frozen=True)
class HypothesisRegistry:
    hypotheses: tuple[Hypothesis, ...]
    sealed_at: str
    schema_version: int = 1

    def validate(self, test_data_available_at: str | None = None) -> None:
        if self.schema_version != 1 or not self.hypotheses:
            raise ValueError("registry schema or hypothesis set is invalid")
        sealed = _timestamp(self.sealed_at)
        hypothesis_ids: set[str] = set()
        trial_ids: set[str] = set()
        for hypothesis in self.hypotheses:
            hypothesis.validate(test_data_available_at)
            if _timestamp(hypothesis.created_before_test_at) > sealed:
                raise ValueError("hypothesis was created after the registry was sealed")
            if hypothesis.hypothesis_id in hypothesis_ids:
                raise ValueError("duplicate hypothesis_id")
            if hypothesis.trial_id in trial_ids:
                raise ValueError("duplicate trial_id")
            hypothesis_ids.add(hypothesis.hypothesis_id)
            trial_ids.add(hypothesis.trial_id)

    def _content(self) -> dict[str, object]:
        ordered = sorted(self.hypotheses, key=lambda item: (item.family_id, item.trial_id))
        values = []
        for hypothesis in ordered:
            value = asdict(hypothesis)
            value["validation_windows"] = list(hypothesis.validation_windows)
            values.append(value)
        return {
            "schema_version": self.schema_version,
            "sealed_at": self.sealed_at,
            "hypotheses": values,
        }

    def sha256(self) -> str:
        self.validate()
        encoded = json.dumps(
            self._content(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def payload(self) -> dict[str, object]:
        content = self._content()
        content["registry_sha256"] = self.sha256()
        return content

    def families(self) -> dict[str, tuple[Hypothesis, ...]]:
        self.validate()
        grouped: dict[str, list[Hypothesis]] = {}
        for hypothesis in self.hypotheses:
            grouped.setdefault(hypothesis.family_id, []).append(hypothesis)
        return {
            family_id: tuple(sorted(values, key=lambda item: item.trial_id))
            for family_id, values in sorted(grouped.items())
        }

    def write(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(
            json.dumps(self.payload(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
