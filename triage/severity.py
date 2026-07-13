"""
Severity scoring — turns a heterogeneous result payload into one priority.

Each AI service returns a slightly different JSON shape, so this module reads
defensively from all the known fields (`urgency`, per-detection `severity`,
`ai_findings`, confidence) and collapses them into a single 0-100 priority
score + a coarse level used to order the worklist.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Coarse levels, highest first.
PriorityLevel = str  # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFO"

_LEVEL_BASE = {
    "CRITICAL": 90,
    "HIGH": 70,
    "MEDIUM": 45,
    "LOW": 20,
    "INFO": 10,
    "ROUTINE": 10,
    "REJECTED": 8,   # input rejected by the OOD/quality gate — not a finding
    "UNKNOWN": 5,
}

# Arabic urgency strings the services emit, mapped to canonical levels.
_AR_URGENCY = {
    "حرج": "CRITICAL",
    "عالي": "HIGH",
    "مرتفع": "HIGH",
    "متوسط": "MEDIUM",
    "روتيني": "ROUTINE",
    "منخفض": "LOW",
}

# Findings that always escalate, regardless of the reported urgency.
_CRITICAL_FINDINGS = {
    "pneumothorax", "fracture", "كسر", "mass", "edema", "consolidation",
}
_HIGH_FINDINGS = {
    "decay", "تسوس", "caries", "apical periodontitis", "effusion",
    "cardiomegaly", "gingivitis",
}


def _canon_level(value: Any) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip()
    upper = s.upper()
    if upper in _LEVEL_BASE:
        return "ROUTINE" if upper == "ROUTINE" else upper
    return _AR_URGENCY.get(s)


def _iter_severities(result: Dict[str, Any]) -> List[str]:
    """Collect every severity/urgency-ish signal from a result payload."""
    out: List[str] = []
    lvl = _canon_level(result.get("urgency"))
    if lvl:
        out.append(lvl)

    for det in result.get("detections", []) or []:
        if isinstance(det, dict):
            lvl = _canon_level(det.get("severity"))
            if lvl:
                out.append(lvl)

    findings = result.get("ai_findings")
    if isinstance(findings, list):
        for f in findings:
            if isinstance(f, dict):
                lvl = _canon_level(f.get("severity"))
                if lvl:
                    out.append(lvl)
    elif isinstance(findings, dict):
        lvl = _canon_level(findings.get("severity"))
        if lvl:
            out.append(lvl)
    return out


def _finding_text(result: Dict[str, Any]) -> str:
    parts = [str(result.get("summary", ""))]
    for det in result.get("detections", []) or []:
        if isinstance(det, dict):
            parts.append(str(det.get("class_name", "")))
    findings = result.get("ai_findings")
    if isinstance(findings, dict):
        parts.append(str(findings.get("primary_diagnosis", "")))
    elif isinstance(findings, list):
        parts += [str(f.get("finding", "")) for f in findings if isinstance(f, dict)]
    return " ".join(parts).lower()


def summarize_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the audit/worklist summary fields from a result payload."""
    result = result or {}
    detections = result.get("detections", []) or []

    # top finding + confidence
    top_finding = None
    max_conf = None
    for det in detections:
        if isinstance(det, dict):
            c = det.get("confidence")
            if isinstance(c, (int, float)) and (max_conf is None or c > max_conf):
                max_conf, top_finding = float(c), det.get("class_name")

    findings = result.get("ai_findings")
    if top_finding is None and isinstance(findings, dict):
        top_finding = findings.get("primary_diagnosis")
    if top_finding is None:
        top_finding = result.get("predicted_class") or result.get("summary")

    num = len(detections)
    if num == 0 and isinstance(findings, list):
        num = len(findings)

    return {
        "top_finding": (str(top_finding)[:200] if top_finding else None),
        "max_confidence": max_conf,
        "num_findings": num,
    }


def priority_of(result: Optional[Dict[str, Any]], status: str = "completed") -> Tuple[int, PriorityLevel]:
    """
    Return (score 0-100, level) for ordering the worklist.
    Failed jobs get a low-but-visible score so they aren't lost.
    """
    if status != "completed" or not result:
        return 15, "INFO"

    # Inputs the OOD/quality gate refused: not a clinical finding, so park them
    # low (they don't clog the worklist) while staying visible and filterable.
    if result.get("input_rejected"):
        return _LEVEL_BASE["REJECTED"], "REJECTED"

    levels = _iter_severities(result)
    score = max((_LEVEL_BASE.get(l, 5) for l in levels), default=_LEVEL_BASE["ROUTINE"])
    level = max(levels, key=lambda l: _LEVEL_BASE.get(l, 0)) if levels else "ROUTINE"

    # Keyword escalation — catch criticality the urgency field may have missed.
    text = _finding_text(result)
    if any(k in text for k in _CRITICAL_FINDINGS):
        score = max(score, _LEVEL_BASE["CRITICAL"])
        level = "CRITICAL"
    elif any(k in text for k in _HIGH_FINDINGS):
        score = max(score, _LEVEL_BASE["HIGH"])
        if _LEVEL_BASE.get(level, 0) < _LEVEL_BASE["HIGH"]:
            level = "HIGH"

    # Nudge by confidence so confident criticals sort above borderline ones.
    summary = summarize_result(result)
    if summary["max_confidence"]:
        score = min(100, score + int(summary["max_confidence"] * 9))

    if level == "ROUTINE":
        level = "LOW"
    return score, level
