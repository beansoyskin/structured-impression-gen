# -*- coding: utf-8 -*-
"""Deterministically verbalize structured impression facts.

Structured facts remain the source of truth. This layer only realizes their
explicit assertion, location, modifier, and suggestive-of relations as text.
"""
from __future__ import annotations

from dataclasses import dataclass


def _normalize_assertion(raw: str) -> str:
    value = (raw or "").lower().replace("measurement::", "")
    if "absent" in value:
        return "absent"
    if "uncertain" in value:
        return "uncertain"
    if "present" in value:
        return "present"
    raise ValueError(f"unsupported assertion: {raw!r}")


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("head") or value.get("finding") or "").strip()
    return ""


def _suggestive_text(value: object) -> str:
    if not isinstance(value, dict):
        return _as_text(value)
    head = _as_text(value)
    if not head or not value.get("assertion"):
        return head
    assertion = _normalize_assertion(str(value["assertion"]))
    if assertion == "absent":
        return f"absence of {head}"
    if assertion == "uncertain":
        return f"possible {head}"
    return head


def _join(values: list | None, formatter=_as_text) -> str:
    terms = [formatter(value) for value in (values or [])]
    terms = [term for term in terms if term]
    if len(terms) < 2:
        return terms[0] if terms else ""
    return ", ".join(terms[:-1]) + " and " + terms[-1]


def verbalize_fact(fact: dict) -> str:
    """Render one fact without adding or removing clinical relations."""
    head = _as_text(fact.get("head"))
    if not head:
        raise ValueError("fact head must not be empty")

    assertion = _normalize_assertion(str(fact.get("assertion") or ""))
    modifiers = _join(fact.get("modifiers"))
    locations = _join(fact.get("locations"))
    suggestive = _join(fact.get("suggestive_of"), _suggestive_text)

    subject = " ".join(part for part in (modifiers, head) if part)
    if assertion == "absent":
        sentence = f"No {subject}"
    elif assertion == "uncertain":
        sentence = f"Possible {subject}"
    else:
        sentence = subject[:1].upper() + subject[1:]

    if locations:
        sentence += f" in the {locations}"
    if suggestive:
        sentence += f", suggestive of {suggestive}"
    return sentence + "."


def _extract_facts(impression: dict | list[dict]) -> list[dict]:
    if isinstance(impression, list):
        return impression
    facts: list[dict] = []
    for bucket in ("positive", "negative", "uncertain", "other"):
        facts.extend(impression.get(bucket, []) or [])
    return facts


@dataclass(frozen=True)
class VerbalizationResult:
    text: str
    sentences: list[str]
    source_facts: list[dict]


def verbalize_impression(impression: dict | list[dict]) -> VerbalizationResult:
    """Convert verified facts to text while retaining the source-fact trace."""
    facts = _extract_facts(impression)
    sentences = [verbalize_fact(fact) for fact in facts]
    return VerbalizationResult(
        text=" ".join(sentences),
        sentences=sentences,
        source_facts=facts,
    )


def _self_test() -> None:
    facts = [
        {
            "head": "pneumonia",
            "assertion": "definitely present",
            "locations": ["left lower lobe"],
            "modifiers": ["multifocal"],
            "suggestive_of": [],
        },
        {
            "head": "pleural effusion",
            "assertion": "definitely absent",
            "locations": ["right pleural space"],
            "modifiers": [],
            "suggestive_of": [],
        },
        {
            "head": "opacity",
            "assertion": "uncertain",
            "locations": ["right upper lobe"],
            "modifiers": ["focal"],
            "suggestive_of": [{"head": "nodule", "assertion": "uncertain"}],
        },
    ]
    result = verbalize_impression(facts)
    assert result.sentences == [
        "Multifocal pneumonia in the left lower lobe.",
        "No pleural effusion in the right pleural space.",
        "Possible focal opacity in the right upper lobe, suggestive of possible nodule.",
    ]
    assert result.source_facts == facts
    try:
        verbalize_fact({"head": "opacity", "assertion": "unknown"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown assertion must not be verbalized as present")
    print("[OK] verbalizer self-test passed")


if __name__ == "__main__":
    _self_test()
