"""Shared plumbing for OpenAI-compatible VLM calls.

Two JSON extractors live here on purpose — they predate each other in
different stages and have different failure semantics; merging them
would silently change error handling at the call sites:

  * :func:`extract_json_object_lenient` — returns ``{}`` on any parse
    failure.  Used by landmark synthesis (step 11), where a bad VLM
    response just means "fall through to the next candidate".
  * :func:`extract_json_object_strict` — raises ``ValueError`` with the
    parse error.  Used by detection rescue (step 09), where the caller
    retries the VLM call and wants the reason.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Dict


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def extract_json_object_lenient(raw: str) -> Dict:
    """Best-effort ``{...}`` extraction; returns ``{}`` when unparseable."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        body = raw[3:]
        if body.startswith("json"):
            body = body[4:]
        if body.endswith("```"):
            body = body[:-3]
        raw = body.strip()
    first = raw.find("{")
    last = raw.rfind("}")
    if first < 0 or last <= first:
        return {}
    try:
        out = json.loads(raw[first:last + 1])
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_json_object_strict(raw: str) -> Dict:
    """``{...}`` extraction over several candidate slices; raises
    ``ValueError`` (with the last parse error) when nothing parses."""
    raw = (raw or "").strip()
    candidates = [raw]
    if raw.startswith("```"):
        body = raw[3:]
        if body.startswith("json"):
            body = body[4:]
        if body.endswith("```"):
            body = body[:-3]
        candidates.append(body.strip())
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        candidates.append(raw[first:last + 1])
    last_err: Exception = ValueError("empty response")
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            last_err = exc
    raise ValueError(f"could not parse JSON: {last_err}")
