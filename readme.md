# Language-instructed Object Navigation

> 中文版见 [readme_zh.md](readme_zh.md).

Landmark-RxR rollout + dataset-curation pipeline adapted from LION-Bench.
Habitat-Lab's `rgbds_agent` provides `rgb`, `depth`, and `semantic`
observations.

This README walks through the four core stages end-to-end:

1. **Rollout** — drive the agent through every selected episode.
2. **Filter pipeline 00-04** — narrow `(instruction_id, sub_idx)` pairs
   down to the ones worth grounding.
3. **Target instance selection** — fix one MP40 instance id per
   surviving sub-path as the navigation target.
4. **VLM pixel-grounded rescue** — recover coarse MPCAT40 targets
   (`appliances` / `lighting` / …) into fine categories (`stove` /
   `lamp` / …) via YOLO-World; VLM as fallback.
5. **Blacklist rescue + consolidate** — re-ground sub-paths whose
   landmark was too generic (`wall` / `door` / `room`) by picking a
   different referrable instance visible at the sub-path end, then
   stitch every surviving sub-trajectory (original + synthesized) into
   one `dataset.json`.

## Setup

Configs are split in two:

- **`configs/rollout/rollout_landmark_rxr.yaml`** — host-level paths
  (Landmark-RxR JSON, MP3D scenes, connectivity, output base). Edit
  once per machine.
- **`configs/selection/<split>/<exp>.yaml`** — per-experiment customization
  (`expname`, episode list, agent, viz toggles). One file per
  experiment, organized by `split` (e.g. `val_unseen/`); everything
  below uses one of these.

The output folder is `results/{split}_{expname}/`. The example
selection used throughout this README is:

```bash
SEL=configs/selection/val_unseen/one_scene_partial.yaml
# split: val_unseen, expname: partial_one_scene
# scan : X7HyMhZNoso, ~19 instructions
# → results/val_unseen_partial_one_scene/
```

Inside `results/{run}/`, the pipeline's canonical state is a single
`survivor.yaml` at the run root. Per-tool subfolders are split by scan
so a single experiment touching multiple scenes keeps their artifacts
isolated; `filters/` (drop diagnostics + audit) stays cross-scan:

```
results/val_unseen_partial_one_scene/
  survivor.yaml            # canonical post-pipeline state (each
                           # filter stage overwrites this in place)
  rollout_viz/X7HyMhZNoso/...
  rewrite/X7HyMhZNoso/...
  scene_categories/X7HyMhZNoso/...
  partition/X7HyMhZNoso/...
  target_instances/X7HyMhZNoso/...
  filters/                 # per-stage NN_*_dropped.yaml + audit.json
```

## 1. Rollout

Run the agent through every selected episode and record per-step
observations.

```bash
bash scripts/rollout.sh \
    --config    configs/rollout/rollout_landmark_rxr.yaml \
    --selection "$SEL"
```

Writes to `results/val_unseen_partial_one_scene/rollout_viz/X7HyMhZNoso/`:

```
{instruction_id}/{sub_idx:03d}/{step:04d}.png    # RGB+depth+semantic frames
frames.jsonl                                      # per-frame metadata
results.json                                      # per-episode metrics + agg
replay.yaml                                       # exact instruction_ids run
config_used.yaml                                  # effective merged config
```

Note: `sub_idx` is **0-indexed** everywhere — folder names are `000/`,
`001/`, ..., not `001/`, `002/`.

`frames.jsonl` is consumed by the partition stage of the filter
pipeline (step 2) so that partition cut-points reflect actual rollout
geometry rather than the reference path.

## 2. Filter pipeline (steps 00-04)

The filter pipeline narrows `(instruction_id, sub_idx)` pairs down to
the ones worth grounding. Run steps 00-04 first; the final filter
(step 10) runs after target instance selection.

Each stage overwrites a single canonical `results/{run}/survivor.yaml`
that captures the current narrower set. Diagnostics live under
`filters/`:

```
{run}/survivor.yaml                  # selection-compatible survivor
                                     # list; downstream tools auto-merge
                                     # this on top of any --exp argument
{run}/filters/NN_{name}_dropped.yaml # per-stage drops + reasons (debug)
{run}/filters/audit.json             # per-(ep, sub) status across stages
```

The stages are **pure filter** — each one accepts the current survivor
set and emits a smaller one. Vocabulary prep + visibility live in
step 3.

The `--exp` flag accepts either a selection YAML path (e.g.
`configs/selection/val_unseen/one_scene_partial.yaml`) or a bare
`expname` (e.g. `one_scene_partial`). Either way, `survivor.yaml` is
auto-merged so downstream stages always see the latest pipeline state
without the user passing it explicitly.

Execution order:

```text
filter steps 00-04 (record / multi_floor / rewrite / blacklist / partition)
        │
        ▼
target instance selection (step 3, scripts 05-08)
        │
        ▼
VLM pixel-grounded rescue (step 4, script 09)
        │
        ▼
apply rescue back into target_instances.json (step 4, script 10)
        │
        ▼
blacklist rescue: find replacement landmarks (step 5, script 13)
        │
        ▼
consolidate (originals + synthesized) → dataset.json (step 5, script 11)
        │
        ▼
attrition report (step 5, script 12 — runnable anytime)
```

#### 2.0 Snapshot (drops nothing)

```bash
bash scripts/00_record_original.sh --exp "$SEL"
```

Initializes `survivor.yaml` from the seed selection when one doesn't
already exist (won't clobber an in-progress pipeline). Also writes
`filters/00_original_dropped.yaml` (empty) + `filters/audit.json`.

#### 2.1 Cross-floor filter

Drops episodes whose reference path crosses a floor (vertical span >
0.5 m).

```bash
bash scripts/01_filter_multi_floor.sh --exp "$SEL"
```

#### 2.2 LLM rewrite

For every surviving sub-instruction, asks the LLM to produce
`landmark + landmark_category + landmark_instruction + spatial_instruction`
plus a per-component breakdown. Then drops landmarks that are hard to
refer to / ground (blacklist contains words like door, window, …).

```bash
GEMINI_API_KEY=... bash scripts/02_rewrite_subinstruction.sh --exp "$SEL"
bash scripts/03_blacklist_landmark.sh --exp "$SEL"
```

Produces:

```
rewrite/X7HyMhZNoso/{instruction_id}/sub_instructions_rewritten_filtered.json
rewrite/X7HyMhZNoso/landmark_mapping_filtered.json    # mention → [labels]
```

The `landmark_mapping_filtered.json` here is the rewriter's own
per-component guess — sometimes pulling in labels from other scenes'
vocabularies. Step 3b cleans it up before visibility annotation.

#### 2.3 Partition (uses rollout frames)

Splits each surviving sub-path into a spatial segment + a landmark
segment, picking the cut point from `rollout_viz/{scan}/frames.jsonl`
(see geometry note below). Drops sub-paths where partition errored.

```bash
bash scripts/04_partition.sh --exp "$SEL"
```

Produces:

```
partition/X7HyMhZNoso/{instruction_id}/partition.json
partition/X7HyMhZNoso/{instruction_id}/{instruction_id}.png
```

After step 04, `survivor.yaml` holds the sub-path-level survivor set
used as input by target instance selection.

##### Partition geometry

Partition reads `rollout_viz/{scan}/frames.jsonl` when present. It
accumulates signed rollout turn angles (`TURN_RIGHT` positive,
`TURN_LEFT` negative). If `|cumulative turn| >= turn_thresh_deg` before
the forward-distance cutoff, the sub-path is treated as "with turn" and
the partition is placed after another 0.3 m of forward motion from that
threshold-crossing step. Otherwise it is treated as move-forward and
cut after 0.3 m from the start of the rollout sub-path segment. If
rollout frames are missing, it falls back to the reference path using
the same distance threshold.

## 3. Target instance selection

Pick one MP40 instance id per surviving sub-path as the navigation
target. The first two steps prep a clean per-scan vocabulary for the
rewriter's mention → label map, then visibility is annotated, the final
target is chosen. After 3d, continue to step 4 for semantic-granularity
filtering.

```text
list_scene_categories      ← scene vocab cache
        │
        ▼
refine_landmark_mapping    ← LLM, rewrites landmark_mapping_filtered.json
        │
        ▼
list_target_instances      ← enumerate candidate instances + uniqueness tag
        │
        ▼
select_target_instances    ← pick the nearest one to the sub-path end
                             (skipped when uniqueness == unique)
```

### 3a. Cache the scan's object vocabulary

Parses each scan's MP3D `.house` to get the instantiated MPCAT40 category
list, matching the labels used by rollout viz. This is the **only** allowed
vocabulary for the refine step below.

```bash
bash scripts/05_get_object_list.sh --exp "$SEL" --objects_only
```

Writes:

```
scene_categories/X7HyMhZNoso/objects.json
```

### 3b. Refine landmark mapping (LLM, per-scan)

Re-asks the LLM to map every mention to candidates drawn **only** from
that scan's `objects.json`. Overwrites the previous
`landmark_mapping_filtered.json` in place.

```bash
GEMINI_API_KEY=... bash scripts/06_refine_landmark_mapping.sh --exp "$SEL"
```

> **Why `max_tokens` is large**: the model in
> `configs/rewrite/rewrite_subinstructions.yaml` is `gemini-2.5-flash`,
> a thinking model whose internal reasoning is billed against
> `max_tokens`. The remap response is one big JSON over every mention
> in a scan, so we set `max_tokens: 32768` to leave room for both the
> thinking trace and the actual output. With a small budget (e.g.
> 4096) the JSON gets truncated mid-string and the script fails with
> `Unterminated string ... could not parse JSON`. Non-thinking
> alternatives like `gemini-2.0-flash` are no longer available to new
> Gemini API accounts.

### 3c. Enumerate candidate target instances at the partition point

For every surviving `(ep, sub)`, render a 360° semantic panorama at
the **partition point** (the turn node between this sub-path and the
next; usually a virtual `virt:...` node from `partition.json`) and
list every visible MP40 instance whose category matches the landmark.
`uniqueness` is decided from the **count of visible instances at that
vantage point**, not the whole-scene total — that count is what
determines whether downstream selection has a unique target without
extra disambiguation. A per-candidate mask PNG is rendered at the
same pose by default.

```bash
bash scripts/07_list_potential_instances.sh --exp "$SEL"
# skip the per-candidate viz PNGs (faster):
bash scripts/07_list_potential_instances.sh --exp "$SEL" --no_save_viz
# tighten the pixel threshold per instance:
bash scripts/07_list_potential_instances.sh --exp "$SEL" --min_pixel_count 100
```

Reads:
- `survivor.yaml` (auto-merged via `--exp`)
- `rewrite/X7HyMhZNoso/{instruction_id}/sub_instructions_rewritten_filtered.json`
- `rewrite/X7HyMhZNoso/landmark_mapping_filtered.json`
- `partition/X7HyMhZNoso/{ep}/partition.json`

Writes:
- `target_instances/X7HyMhZNoso/target_instances.json` — per (ep, sub):
  `landmark`, `semantic_labels`, `matched_category`,
  `matched_categories`, `matched_by`, `pixel_count`, `pixel_fraction`,
  `candidates[]` (each `{id, category, n_pixels}` plus `viz_path` and
  `viz_visible_pixels` when viz is on), and `uniqueness` ∈ {`unique`,
  `ambiguous`, `not_visible`, `no_match`, `partition_pos_unresolvable`}.
- `target_instances/viz/X7HyMhZNoso/{ep}/sub_{NNN}_cand_{IID}.png` —
  one RGB + semantic panorama per visible candidate, rendered at the
  partition point with a target-mask strip below.

### 3d. Choose the target instance

The selection runs **on the sub-path's final node** (`sub_path_nodes[-1]`,
the last step), not the partition point. This matters when ranking
multiple candidates: the agent is supposed to end up near the target,
so distance is measured from where it stops.

Rule:

- **1 visible instance** → that instance (`view_unique`).
- **>1 visible instances** → the instance whose AABB center is
  closest to the sub-path end point (`view_nearest`). The Habitat
  scene is loaded once per scan and its semantic annotations supply
  every instance's center in habitat coordinates.
- **>1 visible but no instance centers available** → fall back to the
  largest-pixel instance (`view_nearest_fallback`).

```bash
bash scripts/08_get_potential_instance.sh --exp "$SEL"
# list every multi-candidate sub-path with its chosen id + distances:
bash scripts/08_get_potential_instance.sh --exp "$SEL" --print_multi
# lighter .house-only debug image instead of the Habitat render:
bash scripts/08_get_potential_instance.sh --exp "$SEL" --viz_mode topdown
# skip viz altogether:
bash scripts/08_get_potential_instance.sh --exp "$SEL" --no_save_viz
```

Writes:
- `target_instances/X7HyMhZNoso/target_instances.json` — per (ep, sub):
  `target_instance_ids`, `status` (one of the verdicts above),
  `selection_distance` (chosen instance's distance to the sub-path
  end, metres), `candidate_distances` (per-id distance map), plus the
  full `candidates[]` carried over from visibility.
- `target_instances/viz_last_frame/X7HyMhZNoso/{ep}/sub_{NNN}_last.png`
  — the rollout's original last-frame visualization.
- `target_instances/viz_partition/X7HyMhZNoso/{ep}/sub_{NNN}_id_{IID}.png`
  — the partition-point target-mask visualization.
- `target_instances/viz_last_frame_instance/X7HyMhZNoso/{ep}/sub_{NNN}_id_{IID}.png`
  — when Habitat rendering is available, an RGB + semantic panorama at
  the sub-path end node with the chosen instance highlighted.

## 4. VLM pixel-grounded rescue (final step)

Recover MPCAT40-coarse targets (`appliances` / `lighting` / `objects` /
…) into fine categories (`stove` / `refrigerator` / `lamp` / …) via
open-vocabulary detection. Primary path is YOLO-World; the VLM is only
used as a fallback when the detector returns nothing (off by default).

Inputs: 3c's `target_instances/{scan}/target_instances.json` + rollout
`frames.jsonl`. Pipeline: pick sub-paths grounded only to coarse semantic
labels → re-render an RGB and raw semantic panorama at the same pose →
prompt YOLO-World with the landmark phrase plus synonym expansions
(`fridge → {fridge, refrigerator}`, `stove → {stove, oven, cooktop,
range, ...}`) → for each detection (highest score first), query the
semantic buffer with category-aware instance recovery, preferring
instances inside the bbox whose MPCat40 name is the coarse bucket
containing the detector's fine class (e.g. an `appliances` instance when
the detector says `stove`); the first detection that yields a
category-matched instance wins. This can rescue examples with
`target_instance_ids: []` because the instance id comes from the
detection box.

The recorded `category` in the rescue output is the **landmark phrase
from the instruction** (the same word that was passed to YOLO as the
prompt), not the detector's own class name. So `stove` stays `stove`
even when YOLO-World fires the `cooktop` class.

```bash
# Dry run — see which coarse sub-paths will be sent to the detector:
bash scripts/09_vlm_rescue.sh \
    --exp "$SEL" \
    --dry_run

# Run for real (first call auto-downloads ~340MB YOLO-World weights + CLIP):
bash scripts/09_vlm_rescue.sh \
    --exp "$SEL"

# Optional VLM fallback (only invoked when YOLO finds nothing above threshold):
GEMINI_API_KEY=... bash scripts/09_vlm_rescue.sh \
    --exp "$SEL" \
    --enable_vlm_fallback
```

Dependency: `pip install ultralytics` (pulls ultralytics + CLIP; runs on
CUDA in the `lion` env).

Main CLI options:

- `--yolo_model`: default `yolov8l-worldv2.pt`. Use `yolov8x-worldv2.pt`
  for more accuracy / slower, or the `s` / `m` variants for speed.
- `--yolo_conf`: default `0.10`.
- `--yolo_imgsz`: default `1024`, matches panorama width.
- `--yolo_device`: override torch device; omit to let ultralytics pick.
- `--enable_vlm_fallback`: opt-in. Requires `--api_key` / `GEMINI_API_KEY`.
- `--sample_radius` / `--search_radius`: tight vs wide search shells
  used during instance recovery.

Writes:

- `target_instances/{scan}/semantic_rescue_categories.json` — per-scan
  rescue dictionary. Main lookup is `instances["{instance_id}"] ->
  {category, confidence, is_rescuable, semantic_category,
  grounding_method, landmarks, examples, image_paths}`.
  `grounding_method` is `yolo_world` or `vlm_fallback`.
- `target_instances/{scan}/vlm_pixel_grounding/{episode_id}/sub_{NNN}_{rgb,bbox,point,mask}.png`
  — clean RGB, detector bbox, bbox center, and the recovered MP3D
  instance mask overlay.
- `target_instances/{scan}/vlm_pixel_grounding/vlm_pixel_grounding_summary.{json,png}`
  — grounding-result JSON plus a contact sheet (4 thumbnails per row,
  last column is the mask overlay; rows where the detector category
  doesn't match the recovered semantic label are flagged `[MISMATCH]`).

The older mask-based rescue is still available:

```bash
bash scripts/build_semantic_rescue_categories.sh --exp "$SEL" --dry_run
```

It only handles examples that already have `target_instance_ids`,
asking the VLM to name the existing target mask. Useful for auditing an
existing selection, but it cannot rescue examples without an instance
id.

### Apply rescue back into target_instances.json

The rescue step writes a side-car (`semantic_rescue_categories.json`).
Step 10 fans those hits back into the live
`target_instances/{scan}/target_instances.json` so the canonical
target-selection record reflects the rescue:

- A sub-path that was previously `not_visible` (empty
  `target_instance_ids`) and has a rescue hit → fill in the rescued
  `instance_id`, set `status = "rescued"`, record `rescue_landmark`
  and `rescue_category` (= the landmark phrase used as the YOLO prompt).
- A sub-path that already had a target → leave the chosen instance
  untouched, annotate `rescue_landmark` / `rescue_category` /
  `rescue_instance_id` so downstream consumers know the rescue
  confirmed it.

```bash
bash scripts/10_apply_rescue.sh --exp "$SEL"
```

Idempotent: re-running clears stale `rescue_*` annotations and rewrites
them, so running it twice produces the same result. The previous
coarse-label drop filter has been removed entirely — a target that's
coarse-bucket-only and gets no rescue hit is kept as-is rather than
dropped, since the original coarse semantic label is still a valid
grounding. Downstream consumers can decide how to treat such targets
themselves.

## 5. Consolidate surviving sub-trajectories

Reads `survivor.yaml` + per-sub-path artifacts produced by all earlier
stages and stitches them into one record per surviving sub-trajectory:
text (full + sub-split + landmark / spatial), path geometry
(`sub_path_nodes` / `spatial_path` / `landmark_path` / heading /
partition kind), the chosen target instance + whether the landmark was
visible from the partition point, any rescue annotations, and pointers
to viz files. Pure aggregation — no LLM / simulator / detector calls.

```bash
bash scripts/11_consolidate.sh --exp "$SEL"
```

Writes:

- `results/{run}/dataset.json` — top-level JSON list of records, one
  per surviving (scan, instruction_id, sub_idx). Each record carries
  the union of fields from the rewrite, partition, and target_instances
  JSONs plus the dataset-level instruction text, plus a
  `synthesized ∈ {false, true}` field so consumers
  can filter on provenance.

### Blacklist rescue (synthesizing replacement landmarks)

A sibling rescue for sub-paths the **blacklist** filter (`03`) cut —
those where the instruction-derived landmark was too generic ("wall",
"door", "room", "doorway", ...). Instead of re-grounding the original
landmark, this step picks a **different** referrable instance visible
at the sub-path end pose and synthesizes a new sub-instruction.

```bash
bash scripts/13_rescue_blacklist.sh --exp "$SEL"
bash scripts/11_consolidate.sh --exp "$SEL"   # re-run after rescue
```

#### Selection logic ([src/check/rescue_blacklist.py:168-228](src/check/rescue_blacklist.py#L168-L228))

For each dropped sub-path, render a 360° semantic panorama at the
**end pose** (last node of `landmark_path`) and feed the per-instance
pixel counts into `_pick_replacement_landmark`. The function returns
either a chosen instance or `None`.

**Hard filters (each candidate must pass all four):**

1. **Pixel threshold** — visible at the end pose with at least
   `MIN_VISIBLE_PIXELS = 200` pixels in the 360° pano (rejects
   instances that are too far / too occluded to be a usable target).
2. **Category in scene metadata** — the `instance_id` must resolve to
   a category via the cached `.house` parse (safety check; rare
   semantic-sensor / .house mismatches are skipped).
3. **Category not blacklisted** — not in `{wall, floor, ceiling, door,
   window, doorway, railing, blinds, curtain, misc, void}`.
4. **Progressively approached** — `dist(partition, center) -
   dist(end, center) ≥ APPROACH_THRESHOLD_M = 0.5 m`. Confirms the
   agent was actually moving towards this instance along the landmark
   half. Only the two endpoints of the sub-path are sampled — a
   zigzag trajectory could in principle slip through, but sub-paths
   are short enough in practice.

Each survivor records `{instance_id, category, pixel_count,
dist_partition_m, dist_end_m, approach_m}`. A side counter
`cat_visible_count` tallies survivors per category — used by Tier 1
below.

**Tiered selection (each tier narrows the pool; falls back when empty):**

| Tier | Rule | Why |
|---|---|---|
| **1. View-uniqueness** | Keep only candidates whose category has exactly one survivor in the FOV (`cat_visible_count[cat] == 1`). | "Walk to **the** bed" is unambiguous only if one bed is visible at this vantage. |
| **2. Concrete category** | Drop candidates whose category is in `COARSE_BUCKETS = {appliances, objects, furniture, lighting}`. | Coarse buckets aren't good landmark words on their own — "walk to the lighting" reads weird. Prefer a finer category. |
| **3. Closest + most prominent** | Sort ascending by `(dist_end_m, -pixel_count)`. | The target should be near where the agent stops; pixel count breaks ties. We sort by distance-to-end (not by approach magnitude) because the destination, not the biggest mover, defines the landmark. |

The `or pool` fallback (`pool = unique_view or candidates`, `pool =
fine or pool`) means each tier is **best-effort, not mandatory** —
if no candidate satisfies the tighter rule, we use the previous tier's
pool. This avoids returning `None` just because nothing happens to be
unique in the FOV.

The final `chosen` dict gets four extra fields written in:

- `scene_instance_count` — total instances of this category in the whole scan
- `fov_instance_count`   — survivor count of this category at this vantage
- `unique_in_fov`        — `fov_instance_count == 1`
- `unique_in_scene`      — `scene_instance_count == 1`

Both `unique_in_*` flags flow through to `dataset.json`'s
`synthesized_from` block so downstream filtering can use them.

#### Outputs

- `target_instances/{scan}/blacklist_rescue.json` — side-car per scan
  with the replacement landmark, the synthesized sub-instruction
  (currently a simple template `"<spatial>. Walk to a <landmark>."`),
  the new target instance id, and approach / uniqueness stats.

The consolidate step then emits these as additional records in
`dataset.json` with `synthesized = true` and a
`synthesized_from` block carrying the original landmark + drop reason.
