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
