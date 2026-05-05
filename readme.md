# Language-instructed Object Navigation

Landmark-RxR rollout + dataset-curation pipeline adapted from LION-Bench.
Habitat-Lab's `rgbds_agent` provides `rgb`, `depth`, and `semantic`
observations.

This README walks through one concrete experiment end-to-end. Every
output path shown is what you actually get on disk for the example
selection.

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
  scene_categories/X7HyMhZNoso/...
  rewrite/X7HyMhZNoso/...
  partition/X7HyMhZNoso/...
  landmark_visibility/X7HyMhZNoso/...
  target_instances/X7HyMhZNoso/...
  perturb_visibility/X7HyMhZNoso/...
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

```bash
# 0) Snapshot the original selection (drops nothing).
bash scripts/filter.sh 0 --from_yaml "$SEL"

# 1) Drop episodes whose path crosses a floor (vertical span > 1.5 m).
bash scripts/filter.sh 1 --from_yaml "$SEL"

# 2) LLM-rewrite each sub-instruction → landmark + spatial atom +
#    components. Partition uses rollout frames when available, then drops ambiguous /
#    rewriter-errored / partition-errored sub-paths.
GEMINI_API_KEY=... bash scripts/filter.sh 2 --from_yaml "$SEL"

# 3) Drop sub-paths whose landmark can't ground to a concrete MP40
#    object: spatial-only ("hallway"), generic room phrases, all-unknown
#    components, or blacklisted transition words.
bash scripts/filter.sh 3 --from_yaml "$SEL"
```

After stage 2 you also get:

```
rewrite/X7HyMhZNoso/sub_instructions_rewritten_filtered.json
rewrite/X7HyMhZNoso/landmark_mapping_filtered.json
partition/X7HyMhZNoso/{instruction_id}/partition.json
partition/X7HyMhZNoso/{instruction_id}/{instruction_id}.png
```

Partition reads `rollout_viz/{scan}/frames.jsonl` when present. It accumulates
signed rollout turn angles (`TURN_RIGHT` positive, `TURN_LEFT` negative). If
`|cumulative turn| >= turn_thresh_deg` before the forward-distance cutoff, the
sub-path is treated as "with turn" and the partition is placed after another
0.3 m of forward motion from that threshold-crossing step. Otherwise it is
treated as move-forward and cut after 0.3 m from the start of the rollout
sub-path segment. If rollout frames are missing, it falls back to the reference
path using the same distance threshold.

After stage 3, `current.yaml` points to `03_blacklist.yaml` — the final
sub-path-level survivor set.

## 3. Scene object lists

Cache each scan's MP3D `.house` object vocabulary so the LLM remap step
has a fixed candidate list (and you can reuse it for debugging).

```bash
bash scripts/list_scene_categories.sh --from_yaml "$SEL" --objects_only
```

Writes:

```
scene_categories/X7HyMhZNoso/objects.json
```

Drop `--objects_only` if you also want a Habitat semantic-annotation
dump (instance counts per category) at `categories.json`.

## 4. Refine landmark mapping

The rewriter's `landmark_mapping_filtered.json` is built
component-by-component during stage 2 and can include labels the LLM
pulled from other scenes' vocabularies. This step re-asks the LLM to
map every mention to candidates drawn **only from this scan's**
`objects.json`.

Inputs (per scan):
- `rewrite/X7HyMhZNoso/sub_instructions_rewritten_filtered.json` — source of mentions
- `scene_categories/X7HyMhZNoso/objects.json` — allowed label vocabulary

Output (overwrites in place):
- `rewrite/X7HyMhZNoso/landmark_mapping_filtered.json` — `{mention: [labels]}`,
  every label guaranteed to be a verbatim entry of `objects.json`

```bash
GEMINI_API_KEY=... bash scripts/refine_landmark_mapping.sh \
    --from_yaml results/val_unseen_partial_one_scene/filters/current.yaml
```

If `objects.json` is missing for a scan, the tool falls back to live
`.house` parsing (with a WARNING). Pass `--rewrite_config` to override
model / temperature / `max_tokens`; defaults live in
`configs/rewrite/rewrite_subinstructions.yaml`.

## 5. Visibility checks

Four sub-steps, all **non-filtering** — none of them touches
`current.yaml`. They classify, choose, and stress-test the landmark
visibility on the survivor set from step 2.

### 5a. Render visibility from each partition point

For every surviving `(ep, sub)`, render a 360° semantic panorama at the
**partition point** and count distinct MP40 instances of the matched
landmark category.

```bash
bash scripts/annotate_visibility.sh --from_yaml "$SEL"
```

Reads:
- `filters/current.yaml`
- `rewrite/X7HyMhZNoso/sub_instructions_rewritten_filtered.json`
- `rewrite/X7HyMhZNoso/landmark_mapping_filtered.json`
- `partition/X7HyMhZNoso/{ep}/partition.json`

Writes:
- `landmark_visibility/X7HyMhZNoso/visibility.json`

Per sub-path: `status` ∈ {`visible`, `not_visible`, `no_match`,
`partition_pos_unresolvable`, `partition_json_missing`}, plus
`n_instances`, `matched_categories`, `pixel_count`, and a per-instance
`instances[]` list.

### 5b. Summarise unique vs ambiguous visible cases

```bash
bash scripts/count_visible_unique.sh --from_yaml "$SEL"
# inspect non-unique cases:
bash scripts/count_visible_unique.sh --from_yaml "$SEL" --print_non_unique
```

Reads `landmark_visibility/*/visibility.json` (all scans), prints a
status breakdown and a per-scan / per-category summary. No file output.

### 5c. Pick a concrete target instance per sub-path

For visible-unique or visible-dominant cases, fix one MP40 instance id
as the navigation target. The "dominant" rule keeps the largest visible
instance when it is at least `--dominance_ratio` (default 3×) the
second-largest.

```bash
bash scripts/select_target_instances.sh --from_yaml "$SEL"
# tune the ratio:
bash scripts/select_target_instances.sh --from_yaml "$SEL" --dominance_ratio 2.5
```

Writes:
- `target_instances/target_instances.json` — per (ep, sub) chosen
  `target_instance_ids` + verdict (`view_unique` / `view_dominant` /
  `ambiguous` / `visibility:no_match` / `visibility:not_visible`)
- `target_instances/viz/X7HyMhZNoso/{ep}/sub_{NNN}_id_{IID}.png` —
  rollout-style RGB+semantic panel with a target-mask strip below.
  Pass `--viz_mode topdown` for a lightweight `.house`-only debug
  image, or `--no_save_viz` to skip.

### 5d. Robustness check: visibility under start-position perturbation

Stand at the **start of each sub-trajectory** and render 8 panoramas
on a 0.5 m circle around that node. Count how many of those positions
still see one of the chosen `target_instance_ids`, and whether any
**other** visible instance shares that target instance's MP40 category.
This diagnoses both target visibility and category-level uniqueness.

```bash
bash scripts/perturb_visibility.sh --from_yaml "$SEL"
# tune the perturbation:
bash scripts/perturb_visibility.sh --from_yaml "$SEL" --radius 0.3 --n 16
# render at the raw position without snapping to navmesh:
bash scripts/perturb_visibility.sh --from_yaml "$SEL" --no_snap
```

Defaults: `--radius 0.5`, `--n 8`, `--min_pixel_count 50`,
`--min_original_pixel_fraction 0.5`, snapping on, `--save_viz` on.
Target visibility in a perturbed view requires both the absolute pixel
floor and at least 50% of the target's original selected-view pixel
count. Heading is fixed at 0 (panorama is equirectangular — only
position matters).

Each perturbation record includes `target_categories`,
`same_category_hits`, `n_other_same_category`, and `category_unique`.
`category_unique: true` means the target was visible enough and no
non-target instance of the same category was visible above the pixel
threshold from that sampled position.

Reads:
- `filters/current.yaml`
- `target_instances/X7HyMhZNoso/target_instances.json` (auto-falls back
  to the global single file if the per-scan layout isn't in place yet)
- Connectivity DB for the start-node 3-D position

Writes:
- `perturb_visibility/X7HyMhZNoso/summary.json` — scan-level summary
  and an index of per-sub-trajectory JSON files.
- `perturb_visibility/X7HyMhZNoso/{ep}/sub_{NNN}/visibility.json` —
  one file per sub-trajectory with `n_perturbations`, `n_visible`,
  `all_visible`, `n_category_unique`, `all_category_unique`, and the
  per-sample records.
- `perturb_visibility/X7HyMhZNoso/{ep}/sub_{NNN}/k_{KK}.png` — one PNG
  per perturbation, saved next to that sub-trajectory's JSON. It shows
  the 360° RGB panorama with red overlay on target-instance pixels and
  a caption (`k / angle / VIS / px current-required / UNIQ-or-SAME /
  snap`). Skip with `--no_save_viz`; tune width with
  `--panorama_width 1024`.
