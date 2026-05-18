"""LLM-driven per-scene landmark mention → MPCAT40 object label remap.

Refines the cross-episode landmark mapping produced by the rewriter by
re-asking an LLM to map each mention to the actual MP3D ``.house`` object
vocabulary of *its own scene*, rather than relying on the rewriter's
per-component judgement.

Output shape:

    {scan_id: {mention: [matching_labels...]}}

Each mention may map to multiple labels (the scene vocabulary contains
many synonyms — "fridge" → "refrigerator" / "appliance"); downstream
visibility / uniqueness checks broaden their match accordingly.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Sequence


_REMAP_SYSTEM = """\
You are a spatial navigation assistant for an indoor robot.

You receive ONE scene's instantiated MPCAT40 object vocabulary and a list of short
noun-phrase mentions extracted from human navigation instructions about
that scene.

For EACH mention, return all labels from the SCENE OBJECT LIST that the
mention could plausibly refer to. Be STRICT: only match when the label
denotes the SAME concrete object as the mention. Synonyms at the SAME
specificity count ("fridge" ↔ "refrigerator", "couch" ↔ "sofa",
"stairs" ↔ "stairway"/"staircase").

Rules:
  • Pick labels ONLY from the SCENE OBJECT LIST verbatim. Do NOT invent
    labels and do NOT alter casing or spacing.
  • Multiple labels per mention are allowed when several entries fit at
    the same specificity.
  • Hard rule — REJECT coarse-bucket labels for fine-grained mentions.
    The coarse buckets are: "appliances", "furniture", "objects",
    "lighting". Examples that MUST return []:
      - "fridge", "stove", "oven", "microwave" when only "appliances"
        is in the list (no "refrigerator" / "stove" / etc.)
      - "sofa", "chair", "bed", "table" when only "furniture" is in
        the list
      - "lamp", "chandelier" when only "lighting" is in the list
    Only return a coarse-bucket label when the MENTION itself is that
    coarse term (e.g. mention="appliance" → ["appliances"] is OK).
  • If nothing in the list fits at the right specificity, return [].
  • Mentions are lowercase; matching is semantic / case-insensitive.

Respond with ONLY a JSON object (no markdown) mapping each input mention
to its list of matched labels.

Example A (refrigerator IS in the list — synonym match keeps it):
  Mentions: ["fridge", "couch", "stairs", "wugbleflub"]
  Object list: [..., "couch", "refrigerator", "sofa", "stairs",
                "stairway", ...]
  Output:
    {
      "fridge":     ["refrigerator"],
      "couch":      ["sofa", "couch"],
      "stairs":     ["stairs", "stairway"],
      "wugbleflub": []
    }

Example B (only the coarse bucket is available — REJECT the match):
  Mentions: ["fridge", "stove", "lamp", "appliance"]
  Object list: [..., "appliances", "lighting", "furniture", ...]
  Output:
    {
      "fridge":    [],
      "stove":     [],
      "lamp":      [],
      "appliance": ["appliances"]
    }
"""


def _build_remap_message(
    mentions: Sequence[str], object_list: Sequence[str]
) -> str:
    return (
        "SCENE OBJECT LIST:\n"
        f"{json.dumps(list(object_list), ensure_ascii=False)}\n\n"
        f"MENTIONS TO MAP ({len(mentions)} total):\n"
        f"{json.dumps(list(mentions), ensure_ascii=False)}"
    )


def _extract_json_object(raw: str) -> Any:
    """Best-effort JSON-object parser tolerant to LLM output noise.

    Tries (in order):
      1. ``json.loads`` on the raw string.
      2. Strip a leading markdown fence (``\\`\\`\\`json`` / ``\\`\\`\\``) and
         the matching trailing fence if present.
      3. Take the slice between the first ``{`` and the last ``}``.

    Raises ``ValueError`` with the original error text if none succeed.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty response")

    last_err: Exception = ValueError("no attempt")
    candidates: List[str] = [raw]

    if raw.startswith("```"):
        body = raw[3:]
        if body.startswith("json"):
            body = body[4:]
        elif body.startswith("\n"):
            pass
        if body.endswith("```"):
            body = body[:-3]
        candidates.append(body.strip())

    first = raw.find("{")
    last  = raw.rfind("}")
    if first >= 0 and last > first:
        candidates.append(raw[first:last + 1])

    for c in candidates:
        c = c.strip()
        if not c:
            continue
        try:
            return json.loads(c)
        except Exception as exc:
            last_err = exc
    raise ValueError(f"could not parse JSON: {last_err}")


def _call_llm_json(
    client,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_delay: float,
    label: str = "",
) -> Any:
    """Strict-JSON wrapper around the OpenAI-compatible chat endpoint.

    Sends ``response_format={"type": "json_object"}`` so the server
    refuses to emit anything but a single JSON object — eliminates the
    "wrapped in markdown / explanation prose" failure mode.  Falls back
    to the lenient extractor when parsing still fails (some servers
    silently ignore the format hint).  Logs the raw response on the
    final attempt to make truncation issues debuggable.
    """
    last_raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
            except TypeError:
                # Older clients that don't accept response_format.
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
            last_raw = (resp.choices[0].message.content or "").strip()
            return _extract_json_object(last_raw)
        except Exception as exc:
            print(f"  [attempt {attempt}/{max_retries}] {label} error: {exc}")
            if attempt < max_retries:
                time.sleep(retry_delay)
    print(f"  [DEBUG] {label} final raw response (truncated to 1000 chars):")
    print("  " + last_raw[:1000].replace("\n", "\n  "))
    raise RuntimeError(f"LLM failed after {max_retries} retries ({label})")


# Coarse MPCAT40 buckets that must never be returned as a match for a
# fine-grained mention. Mirrors src.process.coarse_labels.DEFAULT_COARSE_LABELS;
# duplicated here to keep landmark_remap importable in places that don't
# pull in coarse_labels.
_COARSE_BUCKETS = {"appliances", "furniture", "objects", "lighting"}


def _filter_coarse_for_fine(mention: str, labels: List[str]) -> List[str]:
    """Drop coarse-bucket labels (``appliances`` / ``furniture`` / ``objects`` /
    ``lighting``) when the mention itself is a fine-grained term.

    Safety net: even with the prompt updated, the LLM occasionally still
    returns ``"appliances"`` for ``"fridge"``. Strip those here so the
    downstream visibility check doesn't get a forced-but-coarse match.
    A mention IS considered coarse when it matches one of the bucket
    names (plural or singular), in which case the coarse label stays.
    """
    m = (mention or "").strip().lower()
    coarse_synonyms = set(_COARSE_BUCKETS)
    coarse_synonyms.update(c[:-1] for c in _COARSE_BUCKETS if c.endswith("s"))
    if m in coarse_synonyms:
        return list(labels)
    return [l for l in labels if (l or "").strip().lower() not in _COARSE_BUCKETS]


def remap_scan_mentions(
    client,
    model: str,
    scan: str,
    mentions: Sequence[str],
    object_list: Sequence[str],
    temperature: float = 0.1,
    max_tokens: int = 8192,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Dict[str, List[str]]:
    """Ask the LLM to map every ``mention`` to candidate ``object_list`` labels.

    Returns ``{mention: [label, ...]}``.  Labels not present in
    ``object_list`` (case-insensitive) are silently dropped — the LLM
    sometimes invents close-but-absent vocabulary, and we only want
    grounded matches downstream.

    Coarse-bucket labels (``appliances`` / ``furniture`` / ``objects`` /
    ``lighting``) are stripped from fine-grained mentions even when the
    LLM returns them, so e.g. ``"fridge"`` maps to ``[]`` in a scene
    whose appliance vocabulary only has the coarse bucket.
    """
    if not mentions:
        return {}

    raw = _call_llm_json(
        client=client, model=model,
        system=_REMAP_SYSTEM,
        user=_build_remap_message(mentions, object_list),
        temperature=temperature, max_tokens=max_tokens,
        max_retries=max_retries, retry_delay=retry_delay,
        label=f"remap[{scan}]",
    )
    if not isinstance(raw, dict):
        return {m: [] for m in mentions}

    obj_set = {s.strip().lower() for s in object_list}

    out: Dict[str, List[str]] = {}
    for mention in mentions:
        labels = raw.get(mention)
        if labels is None:
            labels = raw.get(mention.lower(), [])
        if not isinstance(labels, list):
            labels = []
        cleaned: List[str] = []
        seen: set = set()
        for l in labels:
            s = (l or "").strip() if isinstance(l, str) else ""
            if not s:
                continue
            if s.lower() not in obj_set:
                continue
            if s in seen:
                continue
            cleaned.append(s)
            seen.add(s)
        out[mention] = _filter_coarse_for_fine(mention, cleaned)
    return out


def lookup_mention_labels(
    landmark_mapping: Dict, scan: str, mention: str,
) -> List[str]:
    """Resolve a mention to its candidate labels, format-agnostic.

    Handles both the new per-scan layout
    ``{scan: {mention: [labels]}}`` and the legacy flat layout
    ``{mention: [labels]}`` produced by ``build_landmark_mapping``.
    """
    if not landmark_mapping or not mention:
        return []
    scan_entry = landmark_mapping.get(scan)
    if isinstance(scan_entry, dict):
        val = scan_entry.get(mention)
        return list(val) if isinstance(val, list) else []
    val = landmark_mapping.get(mention)
    return list(val) if isinstance(val, list) else []
