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

Outputs under `{output.base_dir}/{run_name}/`:
- `visibility.json` — per-episode results and a summary
- `visibility/{instruction_id}/sub_NN.png` — 3-panel RGB viz (when `visibility.viz: true`)

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
