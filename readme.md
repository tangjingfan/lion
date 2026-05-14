# Language-instructed Object Navigation

> 中文版见 [readme_zh.md](readme_zh.md).

Landmark-RxR rollout + dataset-curation pipeline adapted from LION-Bench.
Habitat-Lab's `rgbds_agent` provides `rgb`, `depth`, and `semantic`
observations.

This README walks through the three core stages end-to-end:

1. **Rollout** — drive the agent through every selected episode.
2. **Filter pipeline** — narrow `(instruction_id, sub_idx)` pairs down to
   the ones worth grounding.
3. **Target instance selection** — fix one MP40 instance id per
   surviving sub-path as the navigation target.

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

## 2. Filter pipeline

Four stages narrow `(instruction_id, sub_idx)` pairs down to the ones
worth grounding. Each stage writes to `results/{run}/filters/`:

```
NN_{name}.yaml          # selection-compatible survivor list
NN_{name}_dropped.yaml  # what was dropped + why
audit.json              # per-(ep, sub) status across every stage
current.yaml            # symlink to the latest stage's keep file
```

`current.yaml` is what every downstream tool reads via `--from_yaml`.
The four stages are **pure filter** — each one accepts a survivor set
and emits a smaller one. Vocabulary prep + visibility live in step 3.

```bash
CURRENT=results/val_unseen_partial_one_scene/filters/current.yaml
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

After stage 3, `current.yaml` points to `03_partition.yaml` — the final
sub-path-level survivor set used by step 3 below.

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
target. Four steps — the first two prep a clean per-scan vocabulary
for the rewriter's mention → label map, then visibility is annotated
and the final target is chosen.

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
- `target_instances/target_instances.json` — per (ep, sub):
  `target_instance_ids`, `status` (one of the verdicts above),
  `selection_distance` (chosen instance's distance to the sub-path
  end, metres), `candidate_distances` (per-id distance map), plus the
  full `candidates[]` carried over from visibility.
- `target_instances/viz/X7HyMhZNoso/{ep}/sub_{NNN}_id_{IID}.png` —
  RGB + semantic panorama rendered at the sub-path end node, with a
  target-mask strip below highlighting the chosen instance. Each PNG
  is what the agent should see at its last step.
