"""Build / refresh ``configs/landmark_referrable.yaml``.

One LLM call classifies every MPCat40 category into one of three tiers
used by step 11 (rescue_blacklist):

  - too_generic — structural / passage surfaces that don't read as a
                  destination (wall, floor, ceiling, door, window, ...).
  - collective  — coarse buckets that group multiple kinds of things
                  (appliances, furniture, lighting, objects, ...).
                  Step 11 will VLM-refine these per instance.
  - fine        — usable as-is in "walk to a X" (chair, sofa, fridge,
                  stairs, ...).

The output is YAML in the repo (``configs/landmark_referrable.yaml``)
so the classification is reviewable / hand-editable; this script is
the suggested baseline, not the canonical source of truth.

Usage
-----
  bash scripts/build_landmark_referrable.sh
  # or
  python src/check/build_landmark_referrable.py \
      --model gemini-2.5-pro \
      --out configs/landmark_referrable.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.process.rewriter import make_client
from src.process.vlm_common import extract_json_object_strict


# MPCat40 categories (matterport / habitat-mp3d). Keeping the list here
# in code so the script doesn't depend on a particular .house file
# being present.
MPCAT40_CATEGORIES: List[str] = [
    "void", "wall", "floor", "chair", "door", "table", "picture",
    "cabinet", "cushion", "window", "sofa", "bed", "curtain",
    "chest_of_drawers", "plant", "sink", "stairs", "ceiling", "toilet",
    "stool", "towel", "mirror", "tv_monitor", "shower", "column",
    "bathtub", "counter", "fireplace", "lighting", "beam", "railing",
    "shelving", "blinds", "gym_equipment", "seating", "board_panel",
    "furniture", "appliances", "clothes", "objects", "misc",
]


SYSTEM_PROMPT = """\
You classify indoor-scene object categories by their usefulness as a
NAVIGATION LANDMARK in synthesized instructions of the form
"... walk to a X.".

For each category, choose exactly one of three tiers:

  - too_generic : structural / passage / catch-all categories that
                  don't read as destinations. Examples: wall, floor,
                  ceiling, door, window, beam, board_panel, void, misc.
                  Rule of thumb: would "walk to the X" be a strange
                  thing to say to someone? If yes, too_generic.
  - collective  : the category groups many concrete kinds of things
                  under one umbrella (appliances includes fridge,
                  stove, oven, ...; furniture includes sofa, table,
                  chair, ...; lighting includes lamps, chandeliers,
                  sconces). A VLM will look at the specific instance
                  and refine the name later, so use this tier ONLY when
                  the category name itself is too generic.
  - fine        : the category itself is concrete enough — "walk to a
                  sofa" / "walk to a fridge" / "walk to the stairs"
                  all sound natural without further refinement.

Return JSON only, no markdown:

  {"<category>": {"tier": "<too_generic|collective|fine>",
                  "reason": "<short justification>"}, ...}

Cover every category in the user's list, exactly once. Use the exact
category names from the list as JSON keys.
"""


def _build_user_text(categories: List[str]) -> str:
    return (
        "Classify each of these MPCat40 categories:\n\n"
        + "\n".join(f"  - {c}" for c in categories)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-2.5-pro",
                    help="LLM model name (default gemini-2.5-pro).")
    ap.add_argument("--api_key", default=None,
                    help="LLM API key; defaults to GEMINI_API_KEY env var.")
    ap.add_argument("--out", default="configs/landmark_referrable.yaml",
                    help="Where to write the YAML.")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    client = make_client(api_key=args.api_key, model=args.model)
    resp = client.chat.completions.create(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_text(MPCAT40_CATEGORIES)},
        ],
    )
    raw = resp.choices[0].message.content
    parsed = extract_json_object_strict(raw)

    # Order: too_generic block, collective block, fine block — matches
    # the hand-curated YAML's layout so diffs are readable.
    ordering = {"too_generic": 0, "collective": 1, "fine": 2}
    rows = []
    for cat in MPCAT40_CATEGORIES:
        rec = parsed.get(cat) or {}
        tier = rec.get("tier") or "fine"
        rows.append((ordering.get(tier, 2), cat, tier, rec.get("reason") or ""))
    rows.sort()

    lines = [
        "# 3-tier classification of MPCat40 categories for landmark synthesis.",
        "# Regenerated by src/check/build_landmark_referrable.py.",
        "# Review the diff before committing.",
        "",
        "categories:",
    ]
    last_tier = None
    for _, cat, tier, reason in rows:
        if tier != last_tier:
            lines.append(f"  # ── {tier} ──")
            last_tier = tier
        lines.append(f"  {cat}: {tier}  # {reason}".rstrip())

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print(f"  too_generic={sum(1 for r in rows if r[2]=='too_generic')}  "
          f"collective={sum(1 for r in rows if r[2]=='collective')}  "
          f"fine={sum(1 for r in rows if r[2]=='fine')}")


if __name__ == "__main__":
    main()
