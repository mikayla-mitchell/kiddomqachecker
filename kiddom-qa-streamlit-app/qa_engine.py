from __future__ import annotations

import csv
import hashlib
import html as ihtml
import io
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from comment_policy import finalize_training_comments


FINAL_FIELDS = ["issue_id", "status", "comment"]
DETAIL_FIELDS = [
    "issue_id",
    "status",
    "checker",
    "confidence",
    "field",
    "original",
    "proposed",
    "comment",
    "context_before",
    "context_after",
    "breadcrumb",
    "node_label",
    "node_link",
]
REVIEW_FIELDS = ["decision", "review_note", *DETAIL_FIELDS]
VALID_STATUSES = {"approved", "rejected", "needs_change"}

STANDARDS_CODE_RE = re.compile(
    r"[A-Z]{1,4}-[A-Z]{2,5}(\.[A-Za-z0-9]+)+|\b\d{1,2}\.[A-Z]{1,3}(\.[A-Za-z0-9]+)*\b"
)
STANDARDS_CODE_PREFIX_RE = re.compile(r"[A-Z]{2,4}-[A-Z]{2,5}$")
MATH_LABEL_RE = re.compile(r"\b(?:Equation|Card|Function|Expression)\s+[A-Z]\s*$")
MATH_LABEL_AMBIGUOUS_RE = re.compile(r"\n\s*[A-Z]\s*$")
MATH_VOCAB_AFTER_RE = re.compile(
    r"^\s*(?:is|are|represents|denotes)\s+(?:a|the|an)?\s*"
    r"(?:function|variable|constant|independent|dependent|coefficient|"
    r"exponent|ratio|rate|slope)",
    re.I,
)
REPEATED_LETTER_RE = re.compile(r"^(.)\1{2,}$", re.I)
ABBREV_WORD_TAIL_RE = re.compile(
    r"\b(?:mr|mrs|ms|dr|st|jr|sr|vs|etc|i\.e|e\.g|no|ave|blvd|approx|"
    r"dept|est|fig|vol|pp|pg|op|assn|inc|corp|ltd|cf)\.\s*$",
    re.I,
)
ABBREV_INITIALS_TAIL_RE = re.compile(r"(?:[A-Z]\.){2,4}\s*$")
LIST_MARKER_TAIL_RE = re.compile(r"(?:\(?\d+\)|\(?[a-zA-Z]\)|\d+\.)\s*$")
CONTAINER_TAG_RE = re.compile(r"</?(th|td|p|li|tr|table)\b")
TABLE_CELL_TAG_RE = re.compile(r"<(/?)(td|th)\b")

CAP_REJECT_KW = [
    "doi",
    "et al",
    "www.",
    "domain name",
    "xmlns",
    "imaginary unit",
    "mathematical constant",
    "trigonometric function",
    "cosine function",
    "sine function",
    "tangent function",
    "euler",
    "sequence term",
    "sequence label",
    "conventionally lowercase",
    "function notation",
    "standard mathematical notation",
    "radians",
    "greek letter",
    "function name",
]
CAP_FLAG_PATTERNS = [
    r"flagged for review",
    r"flagging for review",
    r"human review",
    r"without additional context",
    r"without further context",
    r"without full context",
    r"it is unclear",
    r"is unclear whether",
    r"uncertain whether",
    r"not entirely clear",
    r"risks changing meaning",
    r"not definitively required",
    r"not clear if",
    r"not clear whether",
]


class QAEngineError(ValueError):
    pass


def _get_attr(block: str, attr: str) -> str | None:
    match = re.search(attr + r'="([^"]*)"', block)
    return match.group(1) if match else None


def _strip_tags_with_map(html_text: str) -> tuple[str, list[int]]:
    plain: list[str] = []
    index_map: list[int] = []
    index = 0
    while index < len(html_text):
        if html_text[index] == "<":
            tag_end = html_text.find(">", index)
            if tag_end == -1:
                break
            index = tag_end + 1
            continue
        plain.append(html_text[index])
        index_map.append(index)
        index += 1
    return "".join(plain), index_map


def _normalize_ws(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    index_map: list[int] = []
    for index, character in enumerate(text):
        if not character.isspace():
            normalized.append(character)
            index_map.append(index)
    return "".join(normalized), index_map


def _find_container_tag(
    original_html: str,
    plain: str,
    plain_index_map: list[int],
    anchor: str,
) -> tuple[str | None, bool]:
    """Locate a finding in original HTML and return its text container."""
    if not anchor:
        return None, False
    normalized_plain, normalized_map = _normalize_ws(plain)
    normalized_anchor, _ = _normalize_ws(anchor)
    normalized_anchor = normalized_anchor.strip()
    if not normalized_anchor:
        return None, False

    position = normalized_plain.find(normalized_anchor)
    if position == -1:
        for length in (60, 40, 25, 15, 8):
            if len(normalized_anchor) > length:
                position = normalized_plain.find(normalized_anchor[:length])
                if position != -1:
                    break
    if position == -1 or position >= len(normalized_map):
        return None, False

    plain_position = normalized_map[position]
    if plain_position >= len(plain_index_map):
        return None, False
    raw_position = plain_index_map[plain_position]
    before = original_html[:raw_position]

    open_td = 0
    open_th = 0
    combined = 0
    last_open_end: int | None = None
    for match in TABLE_CELL_TAG_RE.finditer(before):
        closing = bool(match.group(1))
        tag = match.group(2)
        if tag == "td":
            open_td += -1 if closing else 1
        else:
            open_th += -1 if closing else 1
        if closing:
            combined -= 1
        else:
            if combined <= 0:
                tag_close = original_html.find(">", match.end())
                last_open_end = tag_close + 1 if tag_close != -1 else match.end()
            combined += 1

    if open_td > 0 or open_th > 0:
        cell_has_prior_text = False
        if last_open_end is not None:
            between = original_html[last_open_end:raw_position]
            between_plain, _ = _strip_tags_with_map(between)
            cell_has_prior_text = bool(between_plain.strip())
        return ("td" if open_td > 0 else "th"), cell_has_prior_text

    matches = list(CONTAINER_TAG_RE.finditer(before))
    if not matches:
        return None, False
    last = matches[-1]
    if before[last.start() : last.start() + 2] == "</":
        return None, False
    return last.group(1), False


def parse_report_html(html_text: str) -> list[dict[str, Any]]:
    """Parse a Kiddom Issue Annotation Report into unique atomic findings."""
    card_splits = re.split(
        r'(?=<div class="issue-card" id="annotation-card-\d+")', html_text
    )
    cards = [card for card in card_splits if card.startswith('<div class="issue-card"')]

    records: list[dict[str, Any]] = []
    for card in cards:
        card_checker = _get_attr(card, "data-checker")
        card_confidence = _get_attr(card, "data-confidence")
        card_field = _get_attr(card, "data-field")

        node_match = re.search(r"<strong>([^<]+)</strong>:\s*([^&<]+)", card)
        node_type = node_match.group(1).strip() if node_match else None
        node_label = node_match.group(2).strip() if node_match else None
        link_match = re.search(r'href="([^"]+)"', card)
        node_link = link_match.group(1) if link_match else None
        breadcrumb = " > ".join(re.findall(r'breadcrumb-pill">([^<]*)</span>', card))

        text_diff_match = re.search(
            r'data-tab="text-diff">(.*?)</div>\s*</div>', card, re.S
        )
        text_diff_html = text_diff_match.group(1) if text_diff_match else ""
        diff_pairs = list(
            re.finditer(
                r'<span class="inline-del">.*?</span>\s*'
                r'<span class="inline-ins">.*?</span>',
                text_diff_html,
                re.S,
            )
        )

        original_tab_match = re.search(
            r'data-tab="original">(.*?)</div>\s*</div>', card, re.S
        )
        original_html = (
            ihtml.unescape(ihtml.unescape(original_tab_match.group(1)))
            if original_tab_match
            else ""
        )
        if original_html:
            container_plain, container_index_map = _strip_tags_with_map(original_html)
        else:
            container_plain, container_index_map = "", []

        issue_splits = re.split(r'(?=<div class="aggregated-issue")', card)
        issue_chunks = [
            chunk
            for chunk in issue_splits
            if chunk.startswith('<div class="aggregated-issue"')
        ]
        for index, issue_chunk in enumerate(issue_chunks):
            checker = _get_attr(issue_chunk, "data-checker") or card_checker
            confidence = (
                _get_attr(issue_chunk, "data-confidence") or card_confidence
            )
            field = _get_attr(issue_chunk, "data-field") or card_field

            original_match = re.search(
                r'issue-original">(.*?)</span>', issue_chunk, re.S
            )
            proposed_match = re.search(
                r'issue-proposed">(.*?)</span>', issue_chunk, re.S
            )
            original = (
                ihtml.unescape(re.sub("<[^>]+>", "", original_match.group(1)))
                if original_match
                else None
            )
            proposed = (
                ihtml.unescape(re.sub("<[^>]+>", "", proposed_match.group(1)))
                if proposed_match
                else None
            )

            id_match = re.search(r'data-comment-for="([^"]+)"', issue_chunk)
            issue_id = id_match.group(1) if id_match else None
            reasoning_match = re.search(
                r'class="reasoning">(.*?)</div>', issue_chunk, re.S
            )
            reasoning = (
                ihtml.unescape(re.sub("<[^>]+>", "", reasoning_match.group(1))).strip()
                if reasoning_match
                else None
            )

            context_before = ""
            context_after = ""
            if index < len(diff_pairs):
                diff = diff_pairs[index]
                raw_window = 600
                before_raw = text_diff_html[
                    max(0, diff.start() - raw_window) : diff.start()
                ]
                after_raw = text_diff_html[
                    diff.end() : diff.end() + raw_window
                ]
                before_stripped = ihtml.unescape(
                    re.sub("<[^>]+>", "", before_raw)
                )
                after_stripped = ihtml.unescape(
                    re.sub("<[^>]+>", "", after_raw)
                )
                context_before = before_stripped[-150:]
                context_after = after_stripped[:150]

            container_tag = None
            container_cell_has_prior_text = False
            if original_html and original:
                anchor = original + context_after[:60]
                (
                    container_tag,
                    container_cell_has_prior_text,
                ) = _find_container_tag(
                    original_html,
                    container_plain,
                    container_index_map,
                    anchor,
                )

            records.append(
                {
                    "issue_id": issue_id,
                    "checker": checker,
                    "confidence": confidence,
                    "field": field,
                    "node_type": node_type,
                    "node_label": node_label,
                    "node_link": node_link,
                    "breadcrumb": breadcrumb,
                    "original": original,
                    "proposed": proposed,
                    "reasoning": reasoning,
                    "context": f"{context_before} <<DIFF>> {context_after}",
                    "context_before": context_before,
                    "context_after": context_after,
                    "container_tag": container_tag,
                    "container_cell_has_prior_text": container_cell_has_prior_text,
                }
            )

    seen: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    for record in records:
        issue_id = record.get("issue_id")
        if not issue_id or issue_id in seen:
            continue
        seen.add(issue_id)
        deduplicated.append(record)

    if not deduplicated:
        raise QAEngineError(
            "No annotation issues were found. Confirm this is a Kiddom Issue "
            "Annotation Report HTML file."
        )
    return deduplicated


def load_rules(source: str | Path | bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        raw = deepcopy(dict(source))
    elif isinstance(source, bytes):
        raw = json.loads(source.decode("utf-8-sig"))
    else:
        with Path(source).open(encoding="utf-8") as handle:
            raw = json.load(handle)
    return validate_rules(raw)


def validate_rules(raw: Mapping[str, Any]) -> dict[str, Any]:
    rules = deepcopy(dict(raw))
    if rules.get("schema_version") != 1:
        raise QAEngineError("Unsupported rules schema. Expected schema_version 1.")
    protected = rules.get("protected_spelling_terms", [])
    safe_targets = rules.get("safe_typo_targets", [])
    exact_rules = rules.get("exact_rules", [])
    if not isinstance(protected, list) or not all(
        isinstance(item, str) for item in protected
    ):
        raise QAEngineError("protected_spelling_terms must be a list of strings.")
    if not isinstance(safe_targets, list) or not all(
        isinstance(item, str) for item in safe_targets
    ):
        raise QAEngineError("safe_typo_targets must be a list of strings.")
    if not isinstance(exact_rules, list):
        raise QAEngineError("exact_rules must be a list.")
    for rule in exact_rules:
        if not isinstance(rule, dict) or rule.get("status") not in VALID_STATUSES:
            raise QAEngineError("Each exact rule must be an object with a valid status.")
    rules["protected_spelling_terms"] = sorted(
        {item.strip().lower() for item in protected if item.strip()}
    )
    rules["safe_typo_targets"] = sorted(
        {item.strip().lower() for item in safe_targets if item.strip()}
    )
    rules["exact_rules"] = exact_rules
    rules.setdefault("rules_version", "custom")
    return rules


def rules_json_bytes(rules: Mapping[str, Any]) -> bytes:
    normalized = validate_rules(rules)
    return (json.dumps(normalized, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def damerau_levenshtein(left: str, right: str) -> int:
    left, right = left.lower(), right.lower()
    distances: dict[tuple[int, int], int] = {}
    left_len, right_len = len(left), len(right)
    for i in range(-1, left_len + 1):
        distances[(i, -1)] = i + 1
    for j in range(-1, right_len + 1):
        distances[(-1, j)] = j + 1
    for i in range(left_len):
        for j in range(right_len):
            cost = 0 if left[i] == right[j] else 1
            distances[(i, j)] = min(
                distances[(i - 1, j)] + 1,
                distances[(i, j - 1)] + 1,
                distances[(i - 1, j - 1)] + cost,
            )
            if (
                i > 0
                and j > 0
                and left[i] == right[j - 1]
                and left[i - 1] == right[j]
            ):
                distances[(i, j)] = min(
                    distances[(i, j)], distances[(i - 2, j - 2)] + cost
                )
    return distances[(left_len - 1, right_len - 1)]


def _looks_like_standards_or_math(before: str, after: str) -> bool:
    # A dangling backslash means the diff split a LaTeX escape such as "\;"
    # or "\,". A complete expression anywhere else in the lookback window
    # does not make an adjacent prose-spacing fix a math false positive.
    if before.endswith("\\"):
        return True
    if STANDARDS_CODE_RE.search(f"{before} {after[:30]}"):
        return True
    if before and STANDARDS_CODE_PREFIX_RE.search(before):
        return True
    return bool(before and MATH_LABEL_RE.search(before))


def _is_bare_math_token(context_after: str) -> bool:
    """Return whether a flagged letter is an isolated math/label token."""
    direct_letters = re.match(r"[a-zA-Z]*", context_after or "")
    if direct_letters and direct_letters.group(0):
        return False
    rest = (context_after or "").lstrip(" \t\xa0")
    if not rest:
        return True
    next_word = re.match(r"[a-zA-Z]+", rest)
    if next_word:
        return len(next_word.group(0)) <= 1
    return True


def _is_math_vocab_subject(context_after: str) -> bool:
    return bool(MATH_VOCAB_AFTER_RE.match(context_after or ""))


def _classify_capitalization_boundary(context_before: str) -> str:
    """Classify a real sentence boundary, abbreviation, or unclear split."""
    if not context_before.strip():
        return "boundary"
    tail = context_before.rstrip()
    if not tail:
        return "boundary"
    if ABBREV_WORD_TAIL_RE.search(tail) or ABBREV_INITIALS_TAIL_RE.search(tail):
        return "abbreviation"
    if LIST_MARKER_TAIL_RE.search(tail):
        return "boundary"
    if tail[-1] in ".!?":
        return "boundary"
    return "unclear"


def _cap_is_flagged(reasoning: str) -> bool:
    if any(re.search(pattern, reasoning) for pattern in CAP_FLAG_PATTERNS):
        return True
    for match in re.finditer(r"ambiguous", reasoning):
        start = match.start()
        if reasoning[max(0, start - 2) : start] != "un":
            return True
    return False


def _exact_rule_match(
    record: Mapping[str, Any], rules: Mapping[str, Any]
) -> tuple[str, str] | None:
    for rule in rules.get("exact_rules", []):
        matched = True
        for field in ("checker", "field", "original", "proposed"):
            expected = rule.get(field, "*")
            if expected not in (None, "*") and str(record.get(field) or "") != str(
                expected
            ):
                matched = False
                break
        if matched:
            return str(rule["status"]), str(rule.get("comment") or "")
    return None


def _classify_spelling(
    record: Mapping[str, Any],
    confidence_is_informative: bool,
    rules: Mapping[str, Any],
) -> tuple[str, str]:
    original = str(record.get("original") or "").strip()
    original_lower = original.lower()
    proposed = str(record.get("proposed") or "").strip()
    context_before = str(record.get("context_before") or "")

    if context_before.endswith("\\"):
        return (
            "rejected",
            "Bare LaTeX command name; keep the original math notation.",
        )

    if REPEATED_LETTER_RE.fullmatch(original_lower):
        return (
            "rejected",
            "Repeated-letter placeholder or math shorthand, not a misspelling.",
        )

    if confidence_is_informative:
        if record.get("confidence") == "high":
            return "approved", ""
        return (
            "rejected",
            "Appears to be a valid technical/specialized term (or jargon), not "
            "a misspelling — checker false positive; proposed replacement is a "
            "different, incorrect word.",
        )

    if re.fullmatch(r"[a-e]hs[a-z]", original_lower):
        return (
            "rejected",
            "Standards-code boundary artifact, not a misspelling — checker false positive.",
        )
    if original_lower in set(rules["protected_spelling_terms"]):
        return (
            "rejected",
            "Known EdTech/math jargon or abbreviation, not a misspelling — "
            "checker false positive.",
        )
    distance = damerau_levenshtein(original_lower, proposed.lower())
    if distance == 1 and proposed.lower() in set(rules["safe_typo_targets"]):
        return "approved", ""
    return (
        "needs_change",
        "No confidence signal is available in this report. Confirm whether this "
        "is a genuine typo and whether the proposed replacement is correct.",
    )


def classify_record(
    record: Mapping[str, Any],
    confidence_is_informative: bool,
    rules: Mapping[str, Any],
) -> tuple[str, str]:
    exact = _exact_rule_match(record, rules)
    if exact:
        return exact

    checker = str(record.get("checker") or "")
    original = str(record.get("original") or "")
    proposed = str(record.get("proposed") or "")
    reasoning_raw = str(record.get("reasoning") or "")
    reasoning = reasoning_raw.lower()
    context_before = str(record.get("context_before") or "")
    context_after = str(record.get("context_after") or "")

    if checker == "check_spelling":
        return _classify_spelling(record, confidence_is_informative, rules)
    if checker == "check_capitalization":
        if _is_bare_math_token(context_after):
            return (
                "rejected",
                "Bare single-letter math or answer-label token; keep lowercase.",
            )
        if _is_math_vocab_subject(context_after):
            return (
                "rejected",
                "Math variable or function name; keep lowercase.",
            )

        container_tag = record.get("container_tag")
        if container_tag in {"th", "td"}:
            if record.get("container_cell_has_prior_text"):
                boundary = _classify_capitalization_boundary(context_before)
                if boundary == "boundary":
                    return (
                        "approved",
                        "Sentence start inside a prose table cell; capitalize.",
                    )
                if boundary == "abbreviation":
                    return (
                        "rejected",
                        "Text follows an abbreviation inside a table cell; keep lowercase.",
                    )
                return (
                    "needs_change",
                    "Capitalization boundary inside a prose table cell needs context.",
                )
            return (
                "rejected",
                "First text in a curriculum table cell follows the lowercase-label convention.",
            )
        if container_tag in {"p", "li"}:
            boundary = _classify_capitalization_boundary(context_before)
            if boundary == "boundary":
                return (
                    "approved",
                    "Sentence or list-item start; capitalize.",
                )
            if boundary == "abbreviation":
                return (
                    "rejected",
                    "Text follows an abbreviation, not a sentence boundary; keep lowercase.",
                )
            return (
                "needs_change",
                "Capitalization inside paragraph/list text needs context.",
            )
        if (
            "container_tag" in record
            and container_tag is None
            and record.get("field") == "body_value"
        ):
            if re.search(r"['\"‘“]\s*$", context_before):
                return (
                    "approved",
                    "Start of quoted speech; capitalize.",
                )
            return (
                "rejected",
                "Flattened body-value table/list label; keep lowercase.",
            )

        if reasoning_raw:
            if any(keyword in reasoning for keyword in CAP_REJECT_KW):
                return (
                    "rejected",
                    "Math notation, constant, or citation convention is "
                    "conventionally lowercase — checker false positive.",
                )
            if _cap_is_flagged(reasoning):
                return (
                    "needs_change",
                    "Ambiguous whether this is a title/sentence start or math "
                    "notation. Check the surrounding context.",
                )
            return "approved", ""
        if record.get("field") == "title":
            return "approved", ""
        return (
            "needs_change",
            "This report has no reasoning text. Check whether the letter starts "
            "a sentence or is inline math notation.",
        )
    if checker == "check_math":
        if "upstream value" in reasoning_raw or "upstream value" in proposed:
            return (
                "needs_change",
                "Possible upstream numeric inconsistency. Verify against the "
                "full problem before deciding.",
            )
        return (
            "rejected",
            "Math checker misread or mis-evaluated the expression; the proposed "
            "value is unreliable.",
        )
    if checker == "check_proper_nouns":
        combined = f"{context_before} {original} {context_after}".lower()
        if (
            original.lower() == "texas"
            and "essential knowledge and skills" in combined
        ):
            return (
                "approved",
                'Official title "Texas Essential Knowledge and Skills"; capitalize Texas.',
            )
        return "rejected", "Common noun or idiom, not a proper noun."
    if checker == "check_links":
        return "approved", ""
    if checker == "check_punctuation":
        if original == ".." and proposed == ".":
            if context_after.strip():
                return (
                    "rejected",
                    "Sentence-frame trailing-off convention; do not collapse the ellipsis to one period.",
                )
            return (
                "needs_change",
                "End-of-field double period may be a typo or deliberate trailing-off.",
            )
        if original.startswith(".") and proposed.startswith(". ") and len(original) > 2:
            return "approved", ""
        return "needs_change", "Punctuation convention needs human judgment."
    if checker == "check_spacing":
        if re.match(r"^\s[.,?);:]$", original) and re.sub(
            r"^\s+", "", original
        ) == proposed:
            return "approved", ""
        if re.match(r"^,[A-Za-z]$", original) and re.match(
            r"^, [A-Za-z]$", proposed
        ):
            return "approved", ""
        if re.match(r"^\s+$", original) and proposed == " " and len(original) > 1:
            return "approved", ""
        if re.match(r"^[?!][A-Z]$", original):
            return "approved", ""
        if re.match(r"^[.:;][A-Za-z]$", original):
            if _looks_like_standards_or_math(context_before, context_after):
                return (
                    "rejected",
                    "Standards code or math-expression boundary, not a "
                    "missing-space error.",
                )
            if context_before and MATH_LABEL_AMBIGUOUS_RE.search(context_before):
                return (
                    "needs_change",
                    "Bare capital letter after a line break may be answer text "
                    "or a math label. Check what follows.",
                )
            return "approved", ""
        return "needs_change", "Spacing pattern needs human judgment."
    return "needs_change", f'Unrecognized checker "{checker}".'


def classify_records(
    records: Iterable[Mapping[str, Any]], rules: Mapping[str, Any]
) -> list[dict[str, Any]]:
    normalized_rules = validate_rules(rules)
    records_list = [dict(record) for record in records]
    spelling_confidences = {
        record.get("confidence")
        for record in records_list
        if record.get("checker") == "check_spelling"
    }
    confidence_is_informative = len(spelling_confidences) > 1

    rows: list[dict[str, Any]] = []
    for record in records_list:
        status, comment = classify_record(
            record, confidence_is_informative, normalized_rules
        )
        rows.append(
            {
                "issue_id": record.get("issue_id") or "",
                "status": status,
                "comment": comment,
                "checker": record.get("checker") or "",
                "confidence": record.get("confidence") or "",
                "field": record.get("field") or "",
                "original": record.get("original") or "",
                "proposed": record.get("proposed") or "",
                "context_before": record.get("context_before") or "",
                "context_after": record.get("context_after") or "",
                "breadcrumb": record.get("breadcrumb") or "",
                "node_label": record.get("node_label") or "",
                "node_link": record.get("node_link") or "",
            }
        )
    return rows


def parse_decision(raw: str, note: str = "") -> tuple[str | None, str | None]:
    value = (raw or "").strip()
    normalized = value.lower()
    if normalized in {"a", "approve", "approved"}:
        return "approved", note.strip()
    if normalized in {"r", "reject", "rejected"}:
        return "rejected", note.strip()
    if normalized in {"needs_change", "needs change"}:
        return "needs_change", note.strip()
    if not value:
        return None, None
    match = re.match(r"^\s*needs[\s_-]*change\s*[-,:]\s*(.*)$", value, re.I)
    if match:
        return "needs_change", (note.strip() or match.group(1).strip())
    return "needs_change", (note.strip() or value)


def apply_reviews(
    rows: Iterable[Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        review = reviews.get(str(row["issue_id"]), {})
        status, note = parse_decision(
            str(review.get("decision") or ""), str(review.get("review_note") or "")
        )
        if status:
            row["status"] = status
            row["comment"] = note or ""
        if row.get("checker") == "check_links":
            row["status"] = "approved"
            row["comment"] = ""
        merged.append(row)
    return merged


def status_counts(rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("status") or "") for row in rows)


def review_progress(
    initial_rows: Iterable[Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, str]],
) -> tuple[int, int]:
    flagged_ids = [
        str(row["issue_id"])
        for row in initial_rows
        if row.get("status") == "needs_change"
    ]
    completed = sum(
        bool(str(reviews.get(issue_id, {}).get("decision") or "").strip())
        for issue_id in flagged_ids
    )
    return completed, len(flagged_ids)


def _csv_bytes(
    rows: Iterable[Mapping[str, Any]], fields: list[str], quote_all: bool = True
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL,
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def final_csv_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return _csv_bytes(finalize_training_comments(rows), FINAL_FIELDS)


def detailed_csv_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return _csv_bytes(rows, DETAIL_FIELDS)


def review_csv_bytes(
    rows: Iterable[Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, str]],
) -> bytes:
    review_rows = []
    for row in rows:
        if row.get("status") != "needs_change":
            continue
        issue_id = str(row["issue_id"])
        review = reviews.get(issue_id, {})
        review_rows.append(
            {
                "decision": review.get("decision", ""),
                "review_note": review.get("review_note", ""),
                **dict(row),
            }
        )
    return _csv_bytes(review_rows, REVIEW_FIELDS)


def safe_report_name(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_review(?:_detailed)?$", "", stem, flags=re.I)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "kiddom_report"


def report_key(filename: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{safe_report_name(filename)}-{digest}"
