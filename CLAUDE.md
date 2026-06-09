# LION — Language-guided Object Navigation

## What we're building (north star)

The goal is **language-guided object navigation**: an agent follows a
natural-language **guidance prefix** to reach the vicinity of a target, then
performs **object navigation** to find the final object.

We construct these episodes by **reshaping an instruction-following dataset
(Landmark-RxR)**: convert its front portion into a language-guidance prefix,
then **concatenate an object-navigation task at the end**. The curation
pipeline in this repo (rollout → partition → visibility → target selection →
detection rescue → synthesis → consolidate; see `readme.md`) exists to produce
that guidance prefix with high quality.

## Three difficulty tiers (of the language guidance)

1. **Simple** — every step navigates to a **currently-visible** landmark, then
   turns to look at the next landmark, and repeats; the episode ends by going to
   find the object. Every landmark must be a real, **visible**, referrable
   instance at its pose.
2. **Medium** — there are **gaps**: the next target is sometimes **not visible**,
   so the agent must **explore**. The direction to the next step is conveyed by
   the **line connecting two landmarks** (a bearing), not a visible target.
3. **Hard** — the guidance from the simpler tiers may contain **misleading /
   erroneous information**; the agent must be robust to it.

## Current focus

**Simple tier** preparation. This is the lens for current pipeline work: a
"simple" episode requires every sub-path's landmark to be a concrete, visible,
uniquely-referrable MP3D instance. That is exactly why the visibility check
(step 07), target-instance selection (08), detection rescue (09–10), and
landmark synthesis (11) exist — and why sub-paths whose landmark is
`not_visible` / `no_match` / `partition_pos_unresolvable` are dropped or
synthesized rather than kept. When weighing a design choice, prefer the option
that yields clean, always-visible, unambiguous landmarks per step.

## Design rationale: partition & the spatial↔landmark re-stitch

Each Landmark-RxR sub-trajectory is `spatial + landmark` — the spatial describes
the movement that leads **into** that landmark (e.g. "turn right, [reach] the
toilet"). But the object-finding guidance we want is `landmark + spatial`, where
the spatial tells the agent which way to **set off after arriving** at the
landmark, toward the next one — e.g. **"go to the toilet" + "then turn right"**.

So concatenating the guidance requires **re-stitching**: split each sub-trajectory
at the **partition point** into its [spatial turn] and [landmark approach], then
re-pair each landmark with the *next* segment's spatial. The spatial attached to
landmark_N is the turn that launches the agent toward landmark_{N+1}.

This is why visibility is checked **at the partition pose** (step 07): that pose
is where the agent, having reached the previous landmark and executed the spatial
turn, sets off toward this segment's landmark. Confirming the landmark is visible
there validates that the step "go to X, then turn Y → [the next landmark is in
view] → go to it" is actually followable — which is exactly the simple-tier
guarantee. Corollary: the check **must** stay at the partition pose; do NOT
backfill an unresolvable partition pose to the sub-path's end node (at the
destination the landmark is trivially present, which would void the check).
`partition_pos_unresolvable` therefore means the step can't be validated — drop
it, or make the partition-boundary resolution more robust, but keep the check at
the boundary.

The agent uses a **panoramic (equirectangular 360°) camera**, so "visible" means
visible anywhere in the panorama — heading-agnostic by design. The 360° semantic
panorama visibility check is therefore correct and intentional, not a limitation:
if the landmark appears anywhere in the panorama, the agent can see it.

## Pointers

- Pipeline mechanics & stage-by-stage I/O: `readme.md` (`readme_zh.md` 中文).
- Run everything: `scripts/run_all.sh --exp configs/selection/<split>/<exp>.yaml`.
- Env / proxy / model gotchas (conda env, the proxy hang, model id) live in the
  auto-memory under `.claude/.../memory/` — check there before running LLM steps.
