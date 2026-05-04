"""
Sub-instruction spatial rewriter for Landmark-RxR.

Two-step pipeline per episode:
  Step 1 — extract landmark, landmark_category, landmark_instruction,
            spatial_instruction from the sub-instruction text.
  Step 2 — for each non-spatial landmark, decompose the phrase into
            individual components grounded to the scene's MP3D category list.

Output per sub-instruction:
  - landmark              overall landmark phrase
  - landmark_category     "room" | "object" | "spatial"
  - landmark_instruction  "Go to the <landmark>."
  - spatial_instruction   discrete action sequence
  - components            list of {original_mention, semantic_label, description}

Also builds a cross-episode landmark→semantic_label mapping file.

API
---
    from src.process.rewriter import make_client, run_rewriter

    client = make_client(api_key)
    all_results, mapping = run_rewriter(
        episodes, client, scenes_dir,
        model="gemini-2.0-flash", max_workers=4,
    )
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.dataset.landmark_rxr import LandmarkRxREpisode


# ---------------------------------------------------------------------------
#  Client
# ---------------------------------------------------------------------------

def make_client(api_key: str):
    """Return an OpenAI client pointed at Gemini's OpenAI-compatible endpoint."""
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


# ---------------------------------------------------------------------------
#  Scene object list (parsed from MP3D .house file)
# ---------------------------------------------------------------------------

def parse_house_objects(scenes_dir: str, scan: str) -> List[str]:
    """Return sorted list of unique object category names defined in the scene.

    Parses C lines from the MP3D .house file (all defined categories, not just
    instantiated ones) so the LLM has the full vocabulary to match against.
    """
    house_path = Path(scenes_dir) / "mp3d" / scan / f"{scan}.house"
    if not house_path.exists():
        return []

    names: set = set()
    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "C" and len(parts) >= 4:
                try:
                    name = parts[3].replace("#", " ")
                    names.add(name)
                except ValueError:
                    pass

    skip = {"remove", "void", "object", "misc"}
    return [n for n in sorted(names) if n not in skip]


# ---------------------------------------------------------------------------
#  Step 1 — landmark extraction prompt
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
You are a spatial navigation assistant for an indoor robot navigating MP3D scenes.

You receive one or more sub-instruction segments from a human navigation instruction
and the compass heading at the END of each segment.

For EACH segment produce FOUR fields:

  1. LANDMARK
     The specific physical object, place, or spatial feature the agent arrives AT
     by the end of this segment (e.g. "door", "teapot on the table", "corridor",
     "bathroom"). A short noun phrase — no articles, no verbs.

     PREFERENCE — when the segment mentions BOTH a transitional spatial
     feature (doorway, threshold, archway, hallway) AND one or more concrete
     physical objects that are visible / called out at the stopping point
     (bath, table, railing, sofa, fridge, ...), pick the **concrete object**
     as the landmark.  Reasoning: concrete objects map more reliably to the
     scene's MP3D semantic categories and are more distinctive anchors than
     transitional spaces.  Even when the text reads "stop at the doorway",
     if it then says "you can see a bath" or "to the right is a railing",
     prefer the concrete object — the spatial feature is the path, the
     object is the anchor.  Pick the most specific / distinctive object
     when several are mentioned.

  2. LANDMARK_CATEGORY
     Classify the landmark as exactly one of:
       • "object"   — a physical item or fixture (door, mirror, sofa, teapot, stairs)
       • "room"     — a named room or enclosed area (bathroom, kitchen, bedroom)
       • "spatial"  — a transitional spatial feature (doorway, corridor, hallway,
                      archway, landing, threshold, passage)

  3. LANDMARK_INSTRUCTION
     A single sentence: "Go to the <landmark>."

  4. SPATIAL_INSTRUCTION
     The most concise movement cue needed BEFORE reaching the landmark, built
     ONLY from these atoms:
       • "Go forward"  •  "Turn left"  •  "Turn right"  •  "Turn around"
     Hard constraints:
       • At most TWO atoms, separated by ". ". Prefer ONE whenever possible.
       • No repetition ("Go forward. Go forward." is forbidden — collapse to "Go forward.").
       • If a turn is present, drop the trailing "Go forward" (the
         landmark_instruction already implies walking to the landmark).
         Example: "Turn right. Go forward." → "Turn right."
       • If the segment is pure straight motion, output exactly "Go forward."
       • No compass directions, landmark names, quantities, adverbs, or "stop".
     Each atom must be one of the four listed above — nothing else.

  5. KEEP
     true  — the landmark is specific and locatable (a named object, piece of furniture,
              named room, etc. that an agent can reliably navigate to).
     false — the landmark is too ambiguous or too general to locate reliably, e.g.:
               • generic architectural features: "wall", "floor", "ceiling", "area", "space"
               • vague spatial references: "somewhere", "spot", "place", "location"
               • overly broad categories: "somewhere in the room"
             Note: "doorway", "corridor", "hallway" are spatial but still locatable → keep=true.
             Only set keep=false for landmarks a robot genuinely cannot navigate to.

Respond with ONLY a JSON array (no markdown, no explanation), one object per segment,
with keys: "landmark", "landmark_category", "landmark_instruction",
"spatial_instruction", "keep".

Example:
[
  {
    "landmark": "teapot on the table",
    "landmark_category": "object",
    "landmark_instruction": "Go to the teapot on the table.",
    "spatial_instruction": "Turn right.",
    "keep": true
  },
  {
    "landmark": "corridor",
    "landmark_category": "spatial",
    "landmark_instruction": "Go to the corridor.",
    "spatial_instruction": "Turn left.",
    "keep": true
  },
  {
    "landmark": "wall",
    "landmark_category": "object",
    "landmark_instruction": "Go to the wall.",
    "spatial_instruction": "Go forward.",
    "keep": false
  }
]

Example of the PREFERENCE rule (concrete object over spatial feature):

  Sub-instruction:
    "Exit the bedroom and turn left and you'll stop at the next doorway on
    the left, it's a bathroom, you can see a bath. To the right now is just
    the railing, you're done."

  Even though the text says "stop at the next doorway", the segment then
  calls out two concrete objects visible at the stopping point — a bath
  and a railing.  Pick the more distinctive one as the landmark.

  → {
      "landmark": "bath",
      "landmark_category": "object",
      "landmark_instruction": "Go to the bath.",
      "spatial_instruction": "Turn left.",
      "keep": true
    }
"""


def _build_extract_message(episode: LandmarkRxREpisode) -> str:
    lines = [
        "Full navigation instruction (context):",
        f'  "{episode.instruction}"',
        "",
        f"Sub-instruction segments ({len(episode.sub_instructions)} total):",
    ]
    for i, sub_instr in enumerate(episode.sub_instructions):
        heading_rad = episode.headings[i] if i < len(episode.headings) else 0.0
        heading_deg = math.degrees(heading_rad) % 360
        lines.append(f"  [{i}] heading_at_end={heading_deg:.1f}° | {sub_instr}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Step 2 — component decomposition prompt
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """\
You are a spatial navigation assistant for an indoor robot navigating MP3D scenes.

You receive a list of landmark phrases and the scene's object category vocabulary.
For EACH landmark phrase, decompose it into its individual constituent objects or places.

Rules:
  • Only include words/sub-phrases that appear in the landmark phrase itself.
  • A single-word landmark produces exactly ONE component.
  • "teapot on the table" → two components: "teapot", "table".
  • "bathroom door" → two components: "bathroom", "door".
  • Prepositions ("on", "near", "by") are NOT components.

For each component produce a JSON object with:
  • "original_mention"  — the word/phrase exactly as it appears in the landmark phrase
  • "semantic_label"    — the single best-matching label from the SCENE OBJECT LIST
                          (use semantic similarity — "fridge"→"refrigerator", "couch"→"sofa");
                          use "unknown" only if nothing fits
  • "description"       — Extract and paraphrase ONLY what the original sub-instruction
                          says about this object. Stay as close to the source text as
                          possible. Do NOT invent details not present in the text.
                          If the text says nothing specific, write a single generic sentence.

Respond with ONLY a JSON object (no markdown) mapping each input landmark phrase to
its components array.

Example input:
  landmarks: [{"landmark": "teapot on the table", "sub_instruction": "Walk to the small white teapot sitting on the wooden table near the window."}]

Example output:
{
  "teapot on the table": [
    {
      "original_mention": "teapot",
      "semantic_label": "objects",
      "description": "A small white teapot sitting on the wooden table near the window."
    },
    {
      "original_mention": "table",
      "semantic_label": "table",
      "description": "A wooden table near the window."
    }
  ]
}
"""


def _build_decompose_message(
    landmarks: List[Dict[str, str]], object_list: List[str]
) -> str:
    obj_str = ", ".join(object_list) if object_list else "(unavailable)"
    return (
        f"SCENE OBJECT LIST: {obj_str}\n\n"
        f"Landmarks to decompose:\n"
        + json.dumps(landmarks, indent=2, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
#  Shared API helper
# ---------------------------------------------------------------------------

def _call_llm(
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
    """Call the LLM and return parsed JSON (list or dict). Raises on total failure."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)
        except Exception as exc:
            print(f"  [attempt {attempt}/{max_retries}] {label} error: {exc}")
            if attempt < max_retries:
                time.sleep(retry_delay)
    raise RuntimeError(f"LLM failed after {max_retries} retries ({label})")


# ---------------------------------------------------------------------------
#  Fallbacks
# ---------------------------------------------------------------------------

_FALLBACK_COMPONENT = {
    "original_mention": "destination",
    "semantic_label":   "unknown",
    "description":      "No description available.",
}

_FALLBACK_EXTRACT = {
    "landmark":             "destination",
    "landmark_instruction": "Go to the destination.",
    "landmark_category":    "object",
    "spatial_instruction":  "Go forward.",
}

_VALID_CATEGORIES = {"object", "room", "spatial"}


def _validate_components(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        return [dict(_FALLBACK_COMPONENT)]
    out = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        out.append({
            "original_mention": str(c.get("original_mention", "unknown")).strip(),
            "semantic_label":   str(c.get("semantic_label",   "unknown")).strip(),
            "description":      str(c.get("description",      "")).strip(),
        })
    return out or [dict(_FALLBACK_COMPONENT)]


# ---------------------------------------------------------------------------
#  Per-episode processing (two steps)
# ---------------------------------------------------------------------------

def process_episode(
    episode: LandmarkRxREpisode,
    client,
    model: str,
    object_list: List[str],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    filter_ambiguous: bool = False,
) -> Dict[str, Any]:
    """Two-step rewrite for one episode."""
    kw = dict(
        client=client, model=model,
        temperature=temperature, max_tokens=max_tokens,
        max_retries=max_retries, retry_delay=retry_delay,
    )
    ep_label = f"ep={episode.instruction_id}"
    n_subs   = len(episode.sub_instructions)

    # ---- Step 1: extract landmarks ----------------------------------------
    try:
        extractions = _call_llm(
            system=_EXTRACT_SYSTEM,
            user=_build_extract_message(episode),
            label=f"{ep_label} step1",
            **kw,
        )
        if not isinstance(extractions, list):
            raise ValueError("not a list")
        while len(extractions) < n_subs:
            extractions.append(dict(_FALLBACK_EXTRACT))
        extractions = extractions[:n_subs]
    except Exception as exc:
        print(f"  [FAIL step1] {ep_label}: {exc}")
        extractions = [{**_FALLBACK_EXTRACT, "error": "step1 failed"}] * n_subs

    # ---- Step 2: decompose non-spatial landmarks ---------------------------
    # Build {landmark: sub_instruction} map (first occurrence wins for duplicates)
    lm_to_sub: Dict[str, str] = {}
    for i, ex in enumerate(extractions):
        if "error" in ex:
            continue
        cat = str(ex.get("landmark_category", "object")).strip().lower()
        if cat == "spatial":
            continue
        lm = str(ex.get("landmark", "destination"))
        if lm not in lm_to_sub:
            lm_to_sub[lm] = episode.sub_instructions[i] if i < n_subs else ""

    landmarks_to_decompose = [
        {"landmark": lm, "sub_instruction": sub}
        for lm, sub in lm_to_sub.items()
    ]

    decompose_map: Dict[str, List[Dict]] = {}
    if landmarks_to_decompose:
        try:
            raw_map = _call_llm(
                system=_DECOMPOSE_SYSTEM,
                user=_build_decompose_message(landmarks_to_decompose, object_list),
                label=f"{ep_label} step2",
                **kw,
            )
            if isinstance(raw_map, dict):
                for lm, comps in raw_map.items():
                    decompose_map[lm] = _validate_components(comps)
        except Exception as exc:
            print(f"  [FAIL step2] {ep_label}: {exc}")

    # ---- Assemble results --------------------------------------------------
    sub_path_results = []
    for i, sub_instr in enumerate(episode.sub_instructions):
        ex = extractions[i] if i < len(extractions) else {}

        keep = bool(ex.get("keep", True))
        if filter_ambiguous and not keep:
            continue

        category = str(ex.get("landmark_category", "object")).strip().lower()
        if category not in _VALID_CATEGORIES:
            category = "object"

        landmark = str(ex.get("landmark", _FALLBACK_EXTRACT["landmark"]))

        if category == "spatial":
            components = []
        else:
            components = decompose_map.get(landmark, [dict(_FALLBACK_COMPONENT)])

        entry = {
            "sub_idx":              i,
            "original":             sub_instr,
            "landmark":             landmark,
            "landmark_category":    category,
            "landmark_instruction": ex.get("landmark_instruction",
                                           _FALLBACK_EXTRACT["landmark_instruction"]),
            "spatial_instruction":  ex.get("spatial_instruction",
                                           _FALLBACK_EXTRACT["spatial_instruction"]),
            "keep":                 keep,
            "components":           components,
        }
        if "error" in ex:
            entry["error"] = ex["error"]
        sub_path_results.append(entry)

    return {
        "scan":        episode.scan,
        "language":    episode.language,
        "instruction": episode.instruction,
        "sub_paths":   sub_path_results,
    }


# ---------------------------------------------------------------------------
#  Landmark mapping builder
# ---------------------------------------------------------------------------

def build_landmark_mapping(all_results: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Aggregate original_mention → semantic_label across all episodes."""
    mapping: Dict[str, set] = defaultdict(set)
    for ep_result in all_results.values():
        for sub in ep_result.get("sub_paths", []):
            if sub.get("landmark_category") == "spatial":
                continue
            for comp in sub.get("components", []):
                mention = comp.get("original_mention", "").strip().lower()
                label   = comp.get("semantic_label",   "unknown").strip()
                if mention and mention != "unknown":
                    mapping[mention].add(label)
    return {k: sorted(v) for k, v in sorted(mapping.items())}


# ---------------------------------------------------------------------------
#  Pipeline
# ---------------------------------------------------------------------------

def run_rewriter(
    episodes: List[LandmarkRxREpisode],
    client,
    scenes_dir: str,
    model: str = "gemini-2.0-flash",
    max_workers: int = 4,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    filter_ambiguous: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Process all episodes in parallel.

    Returns
    -------
    (all_results, landmark_mapping)
      all_results:      dict keyed by str(instruction_id), sorted numerically
      landmark_mapping: dict of original_mention → [semantic_labels]
    """
    scan_objects: Dict[str, List[str]] = {}
    for ep in episodes:
        if ep.scan not in scan_objects:
            scan_objects[ep.scan] = parse_house_objects(scenes_dir, ep.scan)
            n = len(scan_objects[ep.scan])
            print(f"  [scene] {ep.scan}: {n} object categories loaded")

    all_results: Dict[str, Any] = {}
    n_done = 0

    def _process(ep: LandmarkRxREpisode):
        obj_list = scan_objects.get(ep.scan, [])
        return str(ep.instruction_id), process_episode(
            ep, client, model, obj_list,
            temperature, max_tokens, max_retries, retry_delay,
            filter_ambiguous=filter_ambiguous,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, ep): ep for ep in episodes}
        for fut in as_completed(futures):
            ep = futures[fut]
            try:
                ep_id, result = fut.result()
                all_results[ep_id] = result
                n_done += 1

                n_sub = len(result["sub_paths"])
                n_ok  = sum(1 for s in result["sub_paths"] if "error" not in s)
                print(f"  [{n_done}/{len(episodes)}] ep={ep_id}  "
                      f"scan={result['scan']}  subs={n_ok}/{n_sub}")
                for sub in result["sub_paths"]:
                    tag = "ERR" if "error" in sub else " ok"
                    n_comp = len(sub.get("components", []))
                    print(f"    {tag} [{sub['sub_idx']}] "
                          f"[{sub['landmark_category']}] "
                          f"landmark={sub['landmark']!r}  "
                          f"components={n_comp}")
            except Exception as exc:
                print(f"  [FAIL] ep={ep.instruction_id}: {exc}")

    sorted_results = dict(sorted(all_results.items(), key=lambda kv: int(kv[0])))
    mapping = build_landmark_mapping(sorted_results)
    return sorted_results, mapping
