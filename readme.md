# Language-instructed Object Navigation

This repo contains a Landmark-RxR rollout pipeline adapted from LION-Bench.
The environment uses Habitat-Lab's built-in `rgbds_agent` sensor setup, so each
observation includes `rgb`, `depth`, and `semantic`.

## Run a Rollout

```bash
bash scripts/rollout.sh --config configs/rollout/rollout_landmark_rxr.yaml
```

Before running, update the paths in `configs/rollout/rollout_landmark_rxr.yaml`
for your local Landmark-RxR JSON, MP3D scene directory, and connectivity files.
Semantic observations require semantic-capable scene assets, such as MP3D assets
with semantic annotations. If your scenes only contain geometry, `rgb` and
`depth` can render, but `semantic` will not be meaningful.

## Sub-path Visibility Check

Back-traces a ray between each sub-path's start and end node at eye level to
flag sub-paths whose endpoints cannot see each other.

```bash
bash scripts/visibility.sh --config configs/rollout/rollout_landmark_rxr.yaml
```

Outputs under `{output.base_dir}/{run_name}/visibility/`:
- `visibility.json` — per-episode results and a summary
- `{instruction_id}/sub_NN.png` — 2-panel equirectangular viz (when `visibility.viz: true`)

## Rewrite Sub-instructions

Uses a Gemini LLM to decompose each Landmark-RxR sub-instruction into:
landmark phrase, landmark category (`object` / `room` / `spatial`), a clean
`Go to the <landmark>.` instruction, a movement-only spatial instruction,
and a list of components grounded to the scene's MP3D category vocabulary.

```bash
GEMINI_API_KEY=your_key \
bash scripts/rewrite.sh --config configs/rollout/rollout_landmark_rxr.yaml
```

Tune model / workers / retries in
`configs/rewrite/rewrite_subinstructions.yaml`.

Outputs under `{output.base_dir}/{run_name}/rewrite/`:
- `sub_instructions_rewritten.json` — per-episode rewrite (add `_filtered`
  suffix when `filter: true`)
- `landmark_mapping.json` — cross-episode `original_mention → [semantic_labels]`

## Landmark Uniqueness Check

From each sub-path's end node, counts how many instances of the landmark
category are visible. A sub-path is *unique* if exactly one instance is
visible, *ambiguous* if more than one, *not visible* if zero.

Requires `sub_instructions_rewritten.json` from the rewrite step above
(or override via `uniqueness.rewritten_path` in the rollout config).

```bash
bash scripts/uniqueness.sh --config configs/rollout/rollout_landmark_rxr.yaml
```

Outputs under `{output.base_dir}/{run_name}/obs/`:
- `landmark_uniqueness.json` — per-sub-path verdicts and a summary
- `{instruction_id}/sub_NN.png` — 360° panorama strips (when `uniqueness.render_obs: true`)

## Sub-path Partition Visualization

Partitions each sub-trajectory into a **spatial portion** (completing the
`spatial_instruction` — a turn plus a short forward) and a **landmark
portion** (walking to the landmark). Renders a top-down PNG of the scan's
connectivity graph with the episode's path overlaid, sub-paths
colour-coded, and partition points marked with stars.

Requires a rewritten sub-instruction JSON from the rewrite step above.
Only episodes present in that JSON are rendered.

```bash
bash scripts/partition.sh --config configs/rollout/rollout_landmark_rxr.yaml
# optionally cap the number of episodes:
bash scripts/partition.sh --config configs/rollout/rollout_landmark_rxr.yaml --limit 10
```

Hyper-parameters (`TURN_THRESH_DEG`, `AROUND_THRESH_DEG`,
`FORWARD_MARGIN_NODES`, `FORWARD_DEFAULT_NODES`) live at the top of
`src/process/partition.py`.

Outputs under `{output.base_dir}/{run_name}/partition/`:
- `partition.json` — per-sub-path partition index and diagnostic reason
- `{instruction_id}.png` — top-down graph with partition stars and heading arrows

**Reading the PNG.** Each episode is a grid:

- **Top — overview.** The full trajectory on the episode's bounding box,
  gray line with sub-path index tags `[0]`, `[1]`, … at each segment's
  start. Green ● = start node, red × = goal node.
- **Below — one composite panel per sub-path**, each made of two stacked
  parts:
  1. **Action tape (top strip).** Numbered colored boxes laid out left→right
     for the node sequence `0, 1, …, K`. **Orange = spatial** portion,
     **teal = landmark** portion, **black ring = partition node**. Region
     labels `SPATIAL` / `LANDMARK` sit above the tape; a small tick marks
     "partition" below the boundary node. This strip is the fastest way to
     read `p=<i>/<K>` at a glance.
  2. **2-D map (below the tape).** Top-down `x` vs `−z` (north up) zoomed
     to this sub-path only. Node circles match the tape's colors and
     numbers. Each edge is a role-colored arrow with a small label
     `<distance> m  Δ<deg>°` (edge length and signed heading delta vs the
     start heading). A black arrow on node 0 shows the agent's initial
     facing, labelled `θ₀=<deg>°`. The last node is annotated `→ <landmark>`
     when a landmark is present.

Panel title: `[i] Turn right.  →  stairs    p=2/5  ·  right` (sub-path
index, `spatial_instruction`, landmark, partition index / edge count,
`kind`). Coloring is semantic (orange vs teal) and stable across episodes —
sub-path identity is conveyed only by panel position and the `[i]` tag.
