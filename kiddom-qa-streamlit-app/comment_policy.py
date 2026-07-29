from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


OPERATIONAL_RE = re.compile(
    r"""
    \b(?:
        (?:ai|model|checker)?\s*confidence
        |human\s+(?:review|reviewer|judg(?:e)?ment|check|verification)
        |manual\s+(?:review|check|verification)
        |needs?\s+(?:a\s+)?human
        |flagged?\s+for\s+review
        |needs?\s+(?:further\s+)?review
        |ambiguous|unclear|uncertain
        |insufficient\s+context|not\s+enough\s+context
        |without\s+(?:additional|further|full)\s+context
        |no\s+(?:ai\s+)?reasoning
        |check\s+(?:the|whether|what|against)
        |verify\s+(?:the|whether|what|against)
        |confirm\s+whether
        |possible|may\s+be|could\s+be
        |unrecognized\s+checker
    )\b
    """,
    re.I | re.X,
)
CONCRETE_RE = re.compile(
    r"""
    \b(?:
        false\s+positive
        |misspell(?:ing|ed)?|typo
        |jargon|abbreviation|technical\s+term|specialized\s+term
        |curriculum[-\s]specific\s+(?:term|word|name)
        |valid\s+(?:term|word|name|abbreviation|notation)
        |standards?[-\s]code|markup\s+boundary
        |math\s+checker
        |math(?:ematical)?\s+(?:notation|expression|label|constant|variable)
        |latex|mathml|equation|imaginary\s+unit|math\s+shorthand
        |repeated[-\s]letter|placeholder
        |citation\s+convention|doi|et\s+al
        |proper\s+noun|common\s+noun|idiom
        |missing[-\s]space|spacing\s+error
        |punctuation|ellipsis|period|comma|colon|semicolon
        |capitalization|capitalize|lowercase|uppercase
        |broken\s+link|url
        |original\s+(?:is|was)\s+correct
        |proposed\s+(?:change|replacement|value|word).*
            (?:wrong|incorrect|unreliable)
        |incorrect\s+(?:word|value|replacement)
        |wrong\s+(?:word|value|replacement)
    )\b
    """,
    re.I | re.X,
)
DIRECT_ACTION_RE = re.compile(
    r"\b(?:replace|change|correct|fix|remove|add|insert|delete)\b|\bshould be\b",
    re.I,
)
CONDITIONAL_ACTION_RE = re.compile(r"\b(?:use|keep|retain)\b", re.I)
VAGUE_RE = re.compile(
    r"""
    ^\s*(?:
        approved|rejected|reviewed
        |looks?\s+(?:correct|incorrect|fine|good)
        |(?:no\s+)?change\s+needed
        |no\s+changes?
        |(?:change|correction)\s+needed
        |confirmed\s+by\s+(?:a\s+)?human
    )[\s.!]*$
    """,
    re.I | re.X,
)
REVIEW_PREFIX_RE = re.compile(
    r"""
    ^\s*(?:
        (?:human|manual)\s+review
        |reviewer
        |confirmed\s+by\s+(?:a\s+)?human
    )
    (?:\s+(?:determined|confirmed|found))?
    (?:\s+that)?
    \s*[:\-–—]*\s*
    """,
    re.I | re.X,
)
CLAUSE_SPLIT_RE = re.compile(r"(?:\.\s+|;\s*|\s+(?:--|—|–)\s+)")


def _mentions_change_text(
    clause: str, original: Any = "", proposed: Any = ""
) -> bool:
    normalized = clause.casefold()
    for value in (original, proposed):
        token = str(value or "").strip().casefold()
        if len(token) >= 2 and token in normalized:
            return True
    return False


def _is_training_clause(
    clause: str, original: Any = "", proposed: Any = ""
) -> bool:
    if not clause or VAGUE_RE.fullmatch(clause):
        return False
    if OPERATIONAL_RE.search(clause):
        return False
    if CONCRETE_RE.search(clause) or DIRECT_ACTION_RE.search(clause):
        return True
    if CONDITIONAL_ACTION_RE.search(clause) and (
        _mentions_change_text(clause, original, proposed)
        or re.search(r'["“”\'`][^"“”\'`]+["“”\'`]', clause)
    ):
        return True
    return _mentions_change_text(clause, original, proposed) and bool(
        re.search(r"\b(?:valid|correct|incorrect|wrong|preferred)\b", clause, re.I)
    )


def training_comment(
    comment: Any,
    *,
    status: Any = "",
    checker: Any = "",
    original: Any = "",
    proposed: Any = "",
) -> str:
    del status  # Reserved for future policy versions.
    text = re.sub(r"\s+", " ", str(comment or "")).strip()
    if not text or str(checker or "") == "check_links":
        return ""
    text = REVIEW_PREFIX_RE.sub("", text).strip()
    if not text or VAGUE_RE.fullmatch(text):
        return ""

    clauses = [part.strip(" \t,;") for part in CLAUSE_SPLIT_RE.split(text)]
    useful = [
        clause
        for clause in clauses
        if _is_training_clause(clause, original, proposed)
    ]
    if not useful:
        return ""
    if len(useful) == len(clauses):
        return text
    return "; ".join(useful)


def finalize_training_comments(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    finalized = []
    for source in rows:
        row = dict(source)
        row["comment"] = training_comment(
            row.get("comment"),
            status=row.get("status"),
            checker=row.get("checker"),
            original=row.get("original"),
            proposed=row.get("proposed"),
        )
        finalized.append(row)
    return finalized
