# Language-instructed Object Navigation

> 中文版见 [readme_zh.md](readme_zh.md).

Landmark-RxR rollout + dataset-curation pipeline adapted from LION-Bench.
Habitat-Lab's `rgbds_agent` provides `rgb`, `depth`, and `semantic`
observations.

This README walks through the five core stages end-to-end:

1. **Rollout** — drive the agent through every selected episode.
2. **Filter pipeline 00-04** — record / cross-floor / rewrite /
   blacklist-label / partition. After cross-floor (the only hard
   drop), later stages **label** sub-paths via `survivor.sub_status`
   instead of removing them, so rescue paths can still reach them.
3. **Target instance selection (05-08)** — build a per-scan
   vocabulary, refine the mention→label mapping, annotate visibility
   at the partition pose, and pick one MP40 instance id per
   surviving sub-path.
4. **Detection rescue (09-10)** — for sub-paths whose original
   landmark didn't ground to a fine MPCat40 category, run YOLO-World
   (optionally VLM as fallback) on the partition-pose panorama, then
   fan the hits back into `target_instances.json`.
5. **Landmark synthesis + consolidate (11-13)** — for sub-paths still
   un-grounded (blacklist drops + detection failures), pick a
   **different** referrable instance visible at the partition pose
   and rewrite the instruction. Then stitch every surviving
   sub-trajectory (original + synthesized) into one `dataset.json`
   and print an attrition report.

`scripts/run_all.sh --exp "$SEL"` runs 00→13 in one shot;
`scripts/14_inspection_viz.sh` is an opt-in step that renders one
per-sub PNG annotated with its current pipeline status (read-only,
not in the default chain).

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
the ones worth grounding. Run steps 00-04 first; the rescue / synthesis
stages (09-12) run after target instance selection.

Each stage overwrites a single canonical `results/{run}/survivor.yaml`
that captures the current state. Diagnostics + cross-stage trace live
under `filters/`:

```
{run}/survivor.yaml                  # canonical state — sub_paths +
                                     # sub_status label channel.
                                     # downstream tools auto-merge it
                                     # on top of any --exp argument
{run}/filters/NN_{name}_dropped.yaml # per-stage drops + reasons (debug)
{run}/filters/audit.json             # per-(ep, sub) lifecycle events
                                     # — see "Reading the audit"
```

**Cross-floor is the only hard drop**, and it now runs per sub-path:
each sub-trajectory with `max(y) − min(y) > 0.5 m` is removed from
`survivor.sub_paths`, but the episode survives as long as at least
one sub-path stays single-floor. Past stage 01, every surviving
sub-path stays in `survivor.sub_paths`; later stages **label**
failures into `survivor.sub_status[ep_id][sub_idx]` (e.g.
`"blacklist:llm_keep_false"`, `"partition:rewrite_error"`) rather
than deleting the sub. This keeps labeled drops reachable to the
rescue paths (steps 09 / 11) and to inspection tooling.

The `--exp` flag accepts either a selection YAML path (e.g.
`configs/selection/val_unseen/one_scene_partial.yaml`) or a bare
`expname` (e.g. `one_scene_partial`). Either way, `survivor.yaml` is
auto-merged so downstream stages always see the latest pipeline state
without the user passing it explicitly.

Execution order:

```text
filter 00-04 (record / cross_floor / rewrite / blacklist-label / partition)
        │
        ▼
target instance selection 05-08 (vocab / refine / visibility / select)
        │
        ▼
detection rescue 09 (YOLO-World, optional VLM fallback)
        │
        ▼
apply detection rescue back into target_instances.json (10)
        │
        ▼
landmark synthesis 11 (blacklist drops + detection failures →
                       replacement landmark via voxels + VLM)
        │
        ▼
consolidate (originals + synthesized) → dataset.json (12)
        │
        ▼
attrition report (13 — runnable anytime)
        │
        ▼
inspection viz (14 — opt-in; per-sub status thumbnails)
```

`scripts/run_all.sh --exp "$SEL"` chains 00→13 with the right flags.
`--from NN` / `--to NN` re-runs a sub-range; pass `--dry` to preview.

#### 2.0 Snapshot (drops nothing)

```bash
bash scripts/00_record_original.sh --exp "$SEL"
```

Initializes `survivor.yaml` from the seed selection when one doesn't
already exist (won't clobber an in-progress pipeline). Also writes
`filters/00_original_dropped.yaml` (empty) + `filters/audit.json`.

#### 2.1 Cross-floor filter

Drops sub-paths whose nodes span more than `--threshold_m` (default
0.5 m) vertically. The episode survives if any of its sub-paths stays
single-floor; an episode is dropped wholesale only when every sub-path
crosses floors. The drop yaml records the offending `(ep, sub_idx)`
pairs with their Δy.

```bash
bash scripts/01_filter_multi_floor.sh --exp "$SEL"
# loosen the threshold:
bash scripts/01_filter_multi_floor.sh --exp "$SEL" --threshold_m 0.8
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
the partition is placed after another `forward_distance_m` of forward
motion from that threshold-crossing step. Otherwise it is treated as
move-forward and cut after `forward_distance_m` from the start of the
rollout sub-path segment. If rollout frames are missing, it falls back
to the reference path using the same distance threshold.

`forward_distance_m` defaults to **0.5 m** ([configs/partition/partition.yaml](configs/partition/partition.yaml)).
This is one short step past the turn — far enough that doorframes /
wall corners no longer occlude the destination, but close enough that
the partition pose still represents the "agent reads the instruction"
moment.

#### 2.5 Reading the lifecycle audit

`filters/audit.json` is the single per-(ep, sub) trace covering every
stage from 00 through 12. Each episode entry holds:

```json
"1239": {
  "scan":     "X7HyMhZNoso",
  "language": "en-IN",
  "events":   [{ "stage": "cross_floor", "action": "kept", ... }],
  "verdict":  "kept",
  "sub_paths": {
    "2": {
      "events": [
        {"stage": "blacklist",        "action": "dropped",       "reason": "llm_keep_false"},
        {"stage": "rescue_blacklist", "action": "synthesized",   "new_landmark": "fridge"},
        {"stage": "consolidate",      "action": "included",      "synthesized": true}
      ],
      "verdict":    "synthesized:fridge",
      "in_dataset": true
    }
  }
}
```

Two-field summary first (`verdict` + `in_dataset`); events for the
deep dive. Common queries:

```bash
RUN=results/val_unseen_one_scene_partial

# what's in dataset.json, broken down by verdict
jq -r '.episodes[] | .sub_paths[]
       | select(.in_dataset) | .verdict' "$RUN/filters/audit.json" \
       | sort | uniq -c | sort -rn

# everything the pipeline couldn't keep
jq -r '.episodes | to_entries[] | .key as $ep | .value.sub_paths
       | to_entries[] | select(.value.in_dataset == false)
       | "\($ep)#\(.key)  \(.value.verdict)"' "$RUN/filters/audit.json"

# trace one specific (ep, sub)
jq '.episodes["1239"].sub_paths["2"]' "$RUN/filters/audit.json"
```

`scripts/13_attrition.sh --exp "$SEL"` consolidates the same info into
a per-stage funnel + rescue breakdown — runnable anytime, doesn't
modify any state.

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
# also render per-candidate viz PNGs (slower):
bash scripts/07_list_potential_instances.sh --exp "$SEL" --save_viz
# loosen the pixel threshold per instance (default 2000):
bash scripts/07_list_potential_instances.sh --exp "$SEL" --min_pixel_count 1000
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
  `viz_visible_pixels` when viz is on), and two split fields:
  - `visibility` ∈ {`visible`, `not_visible`, `no_match`,
    `partition_pos_unresolvable`} — whether the landmark is reachable
    from this pose at all.
  - `uniqueness` — `True` / `False` when visible (exactly-1 vs >1
    matching instance), or the string `"not_visible"` mirroring the
    visibility tag so downstream code never has to check visibility
    separately to know there's nothing to pick.
- `target_instances/viz/X7HyMhZNoso/{ep}/sub_{NNN}_cand_{IID}.png` —
  one RGB + semantic panorama per visible candidate, rendered at the
  partition point with a target-mask strip below. **Default off** —
  pass `--save_viz` to opt in.

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

## 4. Detection rescue (step 09)

Recover MPCAT40-coarse targets (`appliances` / `lighting` / `objects` /
…) into fine categories (`stove` / `refrigerator` / `lamp` / …) via
open-vocabulary detection. Primary path is YOLO-World; the VLM is only
used as a fallback when the detector returns nothing (off by default).

Inputs: 3c's `target_instances/{scan}/target_instances.json` + rollout
`frames.jsonl`. Pipeline: pick sub-paths grounded only to coarse
semantic labels (or with no MPCat40 match at all) → re-render RGB +
raw semantic panorama at the partition pose → prompt YOLO-World with
the landmark phrase as a single CLIP-encoded class → take the
top-scored detection's bbox and query the semantic buffer for the
**bbox-majority instance** (the MP3D instance covering the most
pixels inside the box). This rescues examples with empty
`target_instance_ids` since the instance id comes from the detection
box, not the original mapping.

The recorded `category` in the rescue output is the **landmark phrase
from the instruction** (the same word that was passed to YOLO as the
prompt), not the detector's own class name. So `stove` stays `stove`
even when YOLO-World fires the `cooktop` class.

Sub-paths where YOLO finds nothing (or no MP3D instance is recovered
from the bbox) emit a `rescue_failed` event into the lifecycle audit;
step 11 picks them up and tries landmark synthesis instead.

```bash
# Dry run — see which coarse sub-paths will be sent to the detector:
bash scripts/09_detection.sh \
    --exp "$SEL" \
    --dry_run

# Run for real (first call auto-downloads ~340MB YOLO-World weights + CLIP):
bash scripts/09_detection.sh \
    --exp "$SEL"

# Optional VLM fallback (only invoked when YOLO finds nothing above threshold):
GEMINI_API_KEY=... bash scripts/09_detection.sh \
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
- `--save_viz`: opt-in. Writes per-detection debug PNGs (RGB, bbox,
  point, mask) plus a per-scan contact sheet. Default off.

Writes:

- `target_instances/{scan}/semantic_rescue_categories.json` — per-scan
  rescue dictionary. Main lookup is `instances["{instance_id}"] ->
  {category, confidence, is_rescuable, semantic_category,
  grounding_method, landmarks, examples, image_paths}`.
  `grounding_method` is `yolo_world` or `vlm_fallback`.
- `detection/{scan}/{episode_id}/sub_{NNN}_{rgb,bbox,point,mask}.png`
  (only with `--save_viz`) — clean RGB, detector bbox, bbox center, and
  the recovered MP3D instance mask overlay.
- `detection/{scan}/summary.json` — grounding-result JSON. With
  `--save_viz` a contact sheet `summary.png` is rendered alongside it
  (4 thumbnails per row, last column is the mask overlay; rows where
  the detector category doesn't match the recovered semantic label are
  flagged `[MISMATCH]`).

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
bash scripts/12_consolidate.sh --exp "$SEL"
```

Writes:

- `results/{run}/dataset.json` — top-level JSON list of records, one
  per surviving (scan, instruction_id, sub_idx). Each record carries
  the union of fields from the rewrite, partition, and target_instances
  JSONs plus the dataset-level instruction text, plus a
  `synthesized ∈ {false, true}` field so consumers
  can filter on provenance.

### Landmark synthesis (step 11)

A second rescue for sub-paths whose original landmark can't be
grounded. Unlike step 09 (which tries to *find* the original
landmark), this step **replaces** it: pick a different referrable
instance visible at the partition pose and rewrite the
sub-instruction. Records flow into `dataset.json` with
`synthesized = true`.

Three upstream sources feed in, all handled uniformly:

- `origin: "blacklist"` — sub-paths labeled by step 02 because the
  instruction-derived landmark was too generic ("wall", "door",
  "room", "doorway", ...). Read from `filters/02_blacklist_dropped.yaml`.
- `origin: "detection_failure"` — sub-paths where step 09 YOLO / VLM
  couldn't locate the original landmark. Read from the lifecycle
  audit's `detection` / `rescue_failed` events.
- `origin: "visibility_not_visible"` — sub-paths where step 07
  matched the original landmark to a fine MPCat40 category but found
  no instance of it visible at the partition pose (e.g. `"bath"` →
  `"bathtub"` matched, but no bathtub on screen). Read from the
  lifecycle audit's `visibility` events with `visibility ==
  "not_visible"`. Step 12 also excludes the originals for these subs
  — only the synth replacement makes it into `dataset.json` if step
  11 finds one.

```bash
bash scripts/11_rescue_blacklist.sh --exp "$SEL"
bash scripts/12_consolidate.sh --exp "$SEL"   # re-run after rescue
```

#### Pipeline

1. Render an RGB + depth + semantic equirectangular panorama at BOTH
   the partition pose and the end pose.
2. Unproject the equirect depth + semantic into a world-frame 3D point
   cloud per instance (pure numpy — no open3d dependency).
3. Voxelise at `VOXEL_SIZE_M = 0.10 m`; count distinct voxels per
   instance per pose. Voxel counts measure the spatial extent of the
   visible surface, decoupled from how close the agent happens to be —
   cleaner than pixel counts, which conflate size with proximity.
4. Geometric filters (a candidate must pass all):
   - `voxels(partition) / voxels(end) ≥ MIN_VISIBILITY_RATIO = 0.10`
     — at least 10% of the destination view must already be on
     screen at instruction-read time.
   - `dist(partition, center) > dist(end, center)` — agent walks
     toward, not past.
   - Category tier ≠ `too_generic` (see referrability table below).
5. Sort survivors by descending score:

   ```
   score = approach_m × min(1, partition_voxels / SIZE_SATURATION_VOXELS=200)
   ```

   The `size_weight` factor mutes the approach signal when the
   instance occupies few voxels at the instruction-read pose, so a
   tiny speck with a huge approach distance can't outrank a clearly-
   visible nearby landmark.
6. Walk the sorted candidates. Per candidate:
   - **`fine` tier** → use the MPCat40 token directly and commit.
   - **`collective` tier** → render a highlighted instance mask, ask
     the VLM to identify it specifically; accept its free-text answer
     when confidence is `high`, else fall through to the next
     candidate.

   Return the first candidate that yields a usable name. If all
   collective candidates come back `unknown`, emit `rescue_failed`.

#### Referrability table

`configs/landmark_referrable.yaml` classifies every MPCat40 category
into one of three tiers:

| Tier | Categories | Behaviour |
|---|---|---|
| `too_generic` | wall, floor, ceiling, door, window, beam, column, board_panel, void, misc | Never picked. |
| `collective` | appliances, furniture, lighting, objects, clothes, gym_equipment, seating | Picked only when the VLM gives a confident specific name. |
| `fine` | chair, sofa, bed, fridge, stairs, sink, railing, ... | Picked directly using the MPCat40 token. |

The table is committed to the repo and hand-editable. Regenerate via
one LLM call:

```bash
GEMINI_API_KEY=... bash scripts/build_landmark_referrable.sh
```

Review the diff before committing — minor tier flips are normal
across LLM runs.

#### VLM refinement (collective tier)

For each `collective` candidate, an instance-highlight PNG is rendered
at the partition pose and sent to the VLM with:

> "The highlighted region shows a single instance whose coarse
> MPCat40 category is `<bucket>`. Identify the object more
> specifically. Return JSON `{name, confidence, reason}` with name as
> a 1-3 word noun phrase usable in 'walk to a <name>'."

Free-text output — no MPCat40 constraint. Accepted when
`confidence == "high"` AND the name isn't just the bucket token
repeated. Per-instance VLM responses are cached at
`target_instances/<scan>/vlm_instance_labels.json` so repeated runs
over the same scan don't re-pay the cost.

CLI:

- `--no_vlm_refine` — skip the VLM step (collective candidates always
  fall through; only `fine` candidates can win).
- `--vlm_model` — default `gemini-2.5-pro`.
- `--vlm_api_key` — falls back to `GEMINI_API_KEY` env var. With no
  key configured, VLM silently disables itself and pipeline runs as
  if `--no_vlm_refine` was set.

#### Outputs

`target_instances/{scan}/blacklist_rescue.json` — one record per
synthesized sub-path:

```json
{
  "origin":               "blacklist" | "detection_failure",
  "original_landmark":    "<text>",
  "original_reason":      "blacklist:llm_keep_false" | "detection:no_detection" | ...,
  "new_landmark":         "refrigerator",
  "new_landmark_source":  "mpcat40" | "vlm",
  "new_mpcat40":          "appliances",   // underlying coarse category
  "new_instance_id":      42,
  "new_sub_instruction":  "Turn right. Walk to a refrigerator.",
  "spatial_instruction":  "Turn right.",
  "visibility":           "visible",
  "uniqueness":           true,
  "vlm":                  null | {"confidence": "high", "model": "...", "reason": "..."}
}
```

The header records the active hyperparameters
(`voxel_size_m`, `size_saturation_voxels`, `min_visibility_ratio`,
`referrable_table`, `vlm_model`) for traceability.

The consolidate step emits these as additional records in
`dataset.json` with `synthesized = true` and a `synthesized_from`
block carrying `{origin, original_landmark, original_reason}`.
