"""LLM-driven per-scene landmark mention → MP3D object label remap.

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

You receive ONE scene's full MP3D object vocabulary and a list of short
noun-phrase mentions extracted from human navigation instructions about
that scene.

For EACH mention, return all labels from the SCENE OBJECT LIST that the
mention could plausibly refer to.  Synonyms count: "fridge" matches
"refrigerator", "couch" matches "sofa", "stairs" matches "stairs" /
"stairway" / "staircase".

Rules:
  • Pick labels ONLY from the SCENE OBJECT LIST verbatim.  Do NOT invent
    labels and do NOT alter casing or spacing.
  • Multiple labels per mention are allowed when several entries fit.
  • If nothing in the list fits, return an empty list for that mention.
  • Mentions are lowercase; matching is semantic / case-insensitive.

Respond with ONLY a JSON object (no markdown) mapping each input mention
to its list of matched labels.

Example (excerpt object list):
  Mentions: ["fridge", "couch", "stairs", "wugbleflub"]
  Object list: [..., "appliance", "couch", "refrigerator", "sofa",
                "stairs", "stairway", ...]
  Output:
    {
      "fridge":     ["refrigerator", "appliance"],
      "couch":      ["sofa", "couch"],
      "stairs":     ["stairs", "stairway"],
      "wugbleflub": []
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
        out[mention] = cleaned
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
