# Language-instructed Object Navigation

> 中文版见 [readme_zh.md](readme_zh.md).

Landmark-RxR rollout + dataset-curation pipeline adapted from LION-Bench.
Habitat-Lab's `rgbds_agent` provides `rgb`, `depth`, and `semantic`
observations.

This README walks through the four core stages end-to-end:

1. **Rollout** — drive the agent through every selected episode.
2. **Filter pipeline 0-3** — narrow `(instruction_id, sub_idx)` pairs
   down to the ones worth grounding.
3. **Target instance selection** — fix one MP40 instance id per
   surviving sub-path as the navigation target.
4. **Filter pipeline 4 + rescue** — filter final targets by semantic
   granularity and optionally rescue some coarse instances with
   YOLO-World (open-vocab detector; VLM as fallback).

## Setup

Configs are split in two:

- **`configs/rollout/rollout_landmark_rxr.yaml`** — host-level paths
  (Landmark-RxR JSON, MP3D scenes, connectivity, output base). Edit
  once per machine.
- **`configs/selection/<exp>.yaml`** — per-experiment customization
  (`expname`, episode list, agent, viz toggles). One file per
  experiment; everything below uses one of these.

The output folder is `results/{split}_{expname}/`. The example
selection used throughout this README is:

```bash
SEL=configs/selection/one_scene_partial_val_unseen.yaml
# split: val_unseen, expname: partial_one_scene
# scan : X7HyMhZNoso, ~19 instructions
# → results/val_unseen_partial_one_scene/
```

Inside `results/{run}/`, every per-tool subfolder **except `filters/`**
is split by scan, so a single experiment touching multiple scenes keeps
their artifacts isolated:

```
results/val_unseen_partial_one_scene/
  rollout_viz/X7HyMhZNoso/...
  rewrite/X7HyMhZNoso/...
  scene_categories/X7HyMhZNoso/...
  partition/X7HyMhZNoso/...
  target_instances/X7HyMhZNoso/...
  filters/                 # cross-scan; not split
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

## 2. Filter pipeline (stages 0-3)

The filter pipeline narrows `(instruction_id, sub_idx)` pairs down to
the ones worth grounding. Run stages 0-3 first; stage 4 runs after
target instance selection. Each stage writes to `results/{run}/filters/`:

```
NN_{name}.yaml          # selection-compatible survivor list
NN_{name}_dropped.yaml  # what was dropped + why
audit.json              # per-(ep, sub) status across every stage
current.yaml            # symlink to the latest stage's keep file
```

`current.yaml` is what every downstream tool reads via `--from_yaml`.
The stages are **pure filter** — each one accepts a survivor set and
emits a smaller one. Vocabulary prep + visibility live in step 3.

```bash
CURRENT=results/val_unseen_partial_one_scene/filters/current.yaml
```

Note: survivor files such as `filters/current.yaml` or
`03_partition.yaml` go to `--from_yaml`, not `--config`. The `--config`
flag is only for rollout configs such as
`configs/rollout/rollout_landmark_rxr.yaml`.

Execution order:

```text
filter stages 0-3
        │
        ▼
target instance selection (step 3)
        │
        ▼
optional VLM semantic rescue (step 4)
        │
        ▼
filter stage 4 semantic_granularity (step 4)
```

#### 2.0 Snapshot (drops nothing)

```bash
bash scripts/filter.sh 0 --from_yaml "$SEL"
```

Writes `filters/00_snapshot.yaml` and re-points `current.yaml` at it.

#### 2.1 Cross-floor filter

Drops episodes whose reference path crosses a floor (vertical span >
0.5 m).

```bash
bash scripts/filter.sh 1 --from_yaml "$SEL"
```

#### 2.2 LLM rewrite

For every surviving sub-instruction, asks the LLM to produce
`landmark + landmark_category + landmark_instruction + spatial_instruction`
plus a per-component breakdown. Then drops landmarks that are hard to
refer to / ground (blacklist contains words like door, window, …).

```bash
GEMINI_API_KEY=... bash scripts/filter.sh 2 --from_yaml "$SEL"
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
bash scripts/filter.sh 3 --from_yaml "$SEL"
```

Produces:

```
partition/X7HyMhZNoso/{instruction_id}/partition.json
partition/X7HyMhZNoso/{instruction_id}/{instruction_id}.png
```

After stage 3, `current.yaml` points to `03_partition.yaml` — the
sub-path-level survivor set used as input by target instance selection.

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
bash scripts/list_scene_categories.sh --from_yaml "$SEL" --objects_only
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
GEMINI_API_KEY=... bash scripts/refine_landmark_mapping.sh --from_yaml "$SEL"
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
bash scripts/list_target_instances.sh --from_yaml "$SEL"
# skip the per-candidate viz PNGs (faster):
bash scripts/list_target_instances.sh --from_yaml "$SEL" --no_save_viz
# tighten the pixel threshold per instance:
bash scripts/list_target_instances.sh --from_yaml "$SEL" --min_pixel_count 100
```

Reads:
- `filters/current.yaml`
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
bash scripts/select_target_instances.sh --from_yaml "$SEL"
# list every multi-candidate sub-path with its chosen id + distances:
bash scripts/select_target_instances.sh --from_yaml "$SEL" --print_multi
# lighter .house-only debug image instead of the Habitat render:
bash scripts/select_target_instances.sh --from_yaml "$SEL" --viz_mode topdown
# skip viz altogether:
bash scripts/select_target_instances.sh --from_yaml "$SEL" --no_save_viz
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

## 4. Filter pipeline (rescue + stage 4)

### 4a. Pixel-grounded semantic rescue (optional, recommended first)

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

```bash
# Dry run — see which coarse sub-paths will be sent to the detector:
bash scripts/build_vlm_pixel_grounded_rescue.sh \
    --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml \
    --dry_run

# Run for real (first call auto-downloads ~340MB YOLO-World weights + CLIP):
bash scripts/build_vlm_pixel_grounded_rescue.sh \
    --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml

# Optional VLM fallback (only invoked when YOLO finds nothing above threshold):
GEMINI_API_KEY=... bash scripts/build_vlm_pixel_grounded_rescue.sh \
    --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml \
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

Loop back to the filter: stage 4b auto-loads the rescue dict on startup
and flips matching `(episode_id, sub_idx)` or `target_instance_id` hits
to `ok_rescued` in the survivor YAML — no extra wiring needed.

The older mask-based rescue is still available:

```bash
bash scripts/build_semantic_rescue_categories.sh --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml --dry_run
```

It only handles examples that already have `target_instance_ids`,
asking the VLM to name the existing target mask. Useful for auditing an
existing selection, but it cannot rescue examples without an instance
id.

### 4b. Semantic granularity filter (final drop, no LLM)

This step filters semantic-taxonomy failures, not low-visibility cases.
For example, an instruction may say `stove`, but MP3D/MPCAT40 may only
label the selected target instance as the coarse class `appliances`
instead of a standalone `stove` semantic label.

This stage first checks `target_instances/{scan}/semantic_rescue_categories.json`.
If the VLM rescued this `(episode_id, sub_idx)` or the selected instance id
with a finer category, the sub-path is kept and audit records `ok_rescued`;
otherwise the coarse target is dropped.

Rule:

- Read the final selected `target_instance_ids` and inspect the selected
  instance's semantic label.
- Drop it if the selected label is only a configured coarse label
  (default: `appliances`, `objects`, `furniture`, `lighting`) and it was
  not rescued to a finer category.
- Keep it when the selected instance has a more specific semantic label,
  or when VLM rescue succeeded.

```bash
# After rescue, use the stage-3 survivor set as input for the final filter:
bash scripts/filter.sh 4 --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml

# Report counts only, without writing 04_semantic_granularity.yaml or
# moving current.yaml:
python src/check/filter_semantic_granularity.py \
    --config configs/rollout/rollout_landmark_rxr.yaml \
    --from_yaml results/val_unseen_partial_one_scene/filters/03_partition.yaml \
    --report_only
```

Writes:

- `filters/04_semantic_granularity.yaml` — survivor set after semantic
  granularity and rescue checks.
- `filters/04_semantic_granularity_dropped.yaml` — remaining dropped
  examples with reason `coarse_semantic_label`, plus `landmark`,
  `semantic_labels`, `coarse_label`, and `target_instance_ids`.
- `filters/current.yaml` — repointed to `04_semantic_granularity.yaml`.
