"""Remove severity scores and security ratings from tool output."""

from enum import Enum
from typing import Any

# Top-level report fields that imply security posture judgments.
_SCORE_FIELDS = frozenset({
    "overall_risk_score",
    "normalized_risk_score",
    "adversarial_risk_score",
    "shadowlogic_risk_score",
    "shadowlogic_susceptibility_score",
    "impnet_risk_score",
    "extraction_risk_score",
    "privacy_risk_score",
    "structural_score",
    "robustness_score",
    "vulnerability_score",
    "susceptibility_score",
})

# Fields dropped from individual findings and patterns.
_FINDING_RATING_FIELDS = frozenset({
    "severity",
    "cvss_estimate",
    "risk",
    "exploitation_difficulty",
})

# Internal chain registry metadata not shown in export.
_CHAIN_INTERNAL_FIELDS = frozenset({
    "severity_modifiers",
})

# Narrative sections that frame security posture rather than structure.
_NARRATIVE_RATING_FIELDS = frozenset({
    "executive_summary",
    "hardening_recommendations",
    "recommended_defense_points",
})

# ShadowLogic assessment fields that encode severity tiers.
_SHADOWLOGIC_RATING_FIELDS = frozenset({
    "susceptibility_level",
    "susceptibility_score",
    "format_risk",
    "audit_complexity_risk",
    "parameter_hiding_risk",
    "camouflage_risk",
    "integrity_risk",
})


def serialize_for_export(obj: Any) -> Any:
    """Recursively serialize a report object, omitting security ratings."""
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {
            k: serialize_for_export(v)
            for k, v in obj.__dict__.items()
            if k not in _SCORE_FIELDS
            and k not in _NARRATIVE_RATING_FIELDS
            and k not in _FINDING_RATING_FIELDS
        }
    if isinstance(obj, dict):
        return _sanitize_dict(obj)
    if isinstance(obj, (list, tuple)):
        return [serialize_for_export(v) for v in obj]
    return obj


def _sanitize_dict(data: dict) -> dict:
    out = {}
    for key, value in data.items():
        if key in _SCORE_FIELDS or key in _NARRATIVE_RATING_FIELDS:
            continue
        if key in _FINDING_RATING_FIELDS or key in _CHAIN_INTERNAL_FIELDS:
            continue
        if key == "shadowlogic_assessment" and isinstance(value, dict):
            out[key] = {
                k: serialize_for_export(v)
                for k, v in value.items()
                if k not in _SHADOWLOGIC_RATING_FIELDS
            }
            continue
        if key == "attributes" and isinstance(value, dict):
            out[key] = {
                k: serialize_for_export(v)
                for k, v in value.items()
                if k != "severity"
            }
            continue
        if key == "gadget_summary" and isinstance(value, dict):
            cleaned = serialize_for_export(value)
            if isinstance(cleaned, dict) and "critical_locations" in cleaned:
                cleaned["notable_locations"] = cleaned.pop("critical_locations")
            out[key] = cleaned
            continue
        out[key] = serialize_for_export(value)
    return out
