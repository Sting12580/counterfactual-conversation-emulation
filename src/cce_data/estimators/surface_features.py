"""Lightweight reward-relevant surface features for action text."""
from __future__ import annotations

import re

import numpy as np


SURFACE_FEATURE_NAMES: list[str] = [
    "word_count",
    "char_count",
    "sentence_count",
    "paragraph_count",
    "avg_word_length",
    "avg_sentence_words",
    "question_mark_count",
    "exclamation_count",
    "comma_count",
    "colon_count",
    "semicolon_count",
    "bullet_count",
    "numbered_list_count",
    "first_person_count",
    "second_person_count",
    "empathy_phrase_count",
    "validation_phrase_count",
    "specificity_marker_count",
    "safety_referral_phrase_count",
    "hedging_phrase_count",
]

_WORD_RE = re.compile(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?\b")
_SENTENCE_END_RE = re.compile(r"[.!?]+")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*]|\u2022)\s+")
_NUMBERED_LIST_RE = re.compile(r"(?m)^\s*\d+[\.)]\s+")
_FIRST_PERSON_RE = re.compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b", re.IGNORECASE)
_SECOND_PERSON_RE = re.compile(r"\b(?:you|your|yours|yourself)\b", re.IGNORECASE)

_EMPATHY_PHRASES = [
    r"\bi'?m sorry\b",
    r"\bsorry\b",
    r"\bthat sounds\b",
    r"\bmust be\b",
    r"\bi understand\b",
    r"\bi hear\b",
    r"\bi can imagine\b",
    r"\bglad you asked\b",
]
_VALIDATION_PHRASES = [
    r"\bvalid\b",
    r"\breasonable\b",
    r"\bmakes sense\b",
    r"\bunderstandable\b",
    r"\bnot alone\b",
    r"\bnormal to feel\b",
    r"\bconcern(?:s)? (?:is|are) warranted\b",
]
_SPECIFICITY_MARKERS = [
    r"\b\d+(?:\.\d+)?\b",
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mmhg|bpm|%|percent)\b",
    r"\b(?:today|tomorrow|yesterday|daily|weekly|monthly)\b",
    r"\b(?:day|days|week|weeks|month|months|year|years|hour|hours)\b",
    r"\b(?:first|second|third|next|previous)\b",
]
_SAFETY_REFERRAL_PHRASES = [
    r"\b911\b",
    r"\bemergency\b",
    r"\ber\b",
    r"\bed\b",
    r"\burgent care\b",
    r"\bseek (?:medical )?(?:attention|care|help)\b",
    r"\bcall (?:your )?(?:doctor|clinician|physician|provider)\b",
    r"\bsee (?:a|your) (?:doctor|clinician|physician|provider|specialist)\b",
    r"\bfollow[- ]?up\b",
    r"\bappointment\b",
    r"\breferral\b",
    r"\bconsult\b",
    r"\bred flag\b",
]
_HEDGING_PHRASES = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\blikely\b",
    r"\bunlikely\b",
    r"\bperhaps\b",
    r"\bgenerally\b",
    r"\busually\b",
    r"\boften\b",
    r"\bseems?\b",
    r"\bappears?\b",
]


def extract_action_surface_features(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Extract deterministic regex/count features from action text."""
    rows = [_features_for_text(text) for text in texts]
    return np.asarray(rows, dtype=np.float32), list(SURFACE_FEATURE_NAMES)


def standardize_behavior_target_features(
    behavior_features: np.ndarray,
    target_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Z-score surface features using behavior-policy moments only."""
    behavior = _as_2d_float("behavior_features", behavior_features)
    target = _as_2d_float("target_features", target_features)
    if behavior.shape[1] != target.shape[1]:
        raise ValueError("behavior_features and target_features dimensions differ.")

    mean = behavior.mean(axis=0)
    std = behavior.std(axis=0)
    scale = np.where(std > 1e-12, std, 1.0)
    standardized_behavior = (behavior - mean) / scale
    standardized_target = (target - mean) / scale
    diagnostics = {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "standardized_on": "behavior",
    }
    return (
        standardized_behavior.astype(np.float32),
        standardized_target.astype(np.float32),
        diagnostics,
    )


def _features_for_text(text: str) -> list[float]:
    value = "" if text is None else str(text)
    lowered = value.lower()
    words = _WORD_RE.findall(value)
    word_lengths = [len(w) for w in words]
    word_count = len(words)
    sentence_count = _sentence_count(value, word_count)
    paragraph_count = _paragraph_count(value, word_count)

    return [
        float(word_count),
        float(len(value)),
        float(sentence_count),
        float(paragraph_count),
        _safe_ratio(sum(word_lengths), word_count),
        _safe_ratio(word_count, sentence_count),
        float(value.count("?")),
        float(value.count("!")),
        float(value.count(",")),
        float(value.count(":")),
        float(value.count(";")),
        float(len(_BULLET_RE.findall(value))),
        float(len(_NUMBERED_LIST_RE.findall(value))),
        float(len(_FIRST_PERSON_RE.findall(value))),
        float(len(_SECOND_PERSON_RE.findall(value))),
        float(_phrase_count(lowered, _EMPATHY_PHRASES)),
        float(_phrase_count(lowered, _VALIDATION_PHRASES)),
        float(_phrase_count(lowered, _SPECIFICITY_MARKERS)),
        float(_phrase_count(lowered, _SAFETY_REFERRAL_PHRASES)),
        float(_phrase_count(lowered, _HEDGING_PHRASES)),
    ]


def _sentence_count(text: str, word_count: int) -> int:
    if word_count == 0:
        return 0
    endings = len(_SENTENCE_END_RE.findall(text))
    return max(1, endings)


def _paragraph_count(text: str, word_count: int) -> int:
    if word_count == 0:
        return 0
    paragraphs = [p for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    return max(1, len(paragraphs))


def _phrase_count(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _as_2d_float(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr
