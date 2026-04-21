"""
Episode visualizer for LION-Bench.

Saves one PNG per step to:
    {out_dir}/viz/{instruction_id}/step_NNNN.png

Layout: two panels side by side, height matches the actual RGB observation.
    ┌──────────────────┬──────────────────┐
    │                  │  scan            │
    │  RGB observation │  instr_id        │
    │  (native size)   │  step / action   │
    │                  │  instruction     │
    └──────────────────┴──────────────────┘

Config (under output.viz):
    enabled:     bool  — master switch       (default: false)
    frame_skip:  int   — save every Nth step (default: 1)
    info_width:  int   — info panel width in px (default: 300)
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from src.process.visibility import VisibilityChecker

# Colours shared by visibility / uniqueness panels
_VIS_BG      = (25,  25,  35)
_VIS_FG      = (220, 220, 220)
_VIS_VISIBLE = (80,  200, 120)
_VIS_BLOCKED = (220, 80,  80)
_VIS_ACCENT  = (100, 200, 255)

from src.dataset.landmark_rxr import LandmarkRxREpisode
from src.env.habitat_env import MOVE_FORWARD, STOP, TURN_LEFT, TURN_RIGHT

_ACTION_LABELS = {
    STOP: "STOP",
    MOVE_FORWARD: "FORWARD",
    TURN_LEFT: "TURN LEFT",
    TURN_RIGHT: "TURN RIGHT",
}

_BG     = (25, 25, 35)       # dark background for info panel
_FG     = (220, 220, 220)    # default text colour
_ACCENT = (100, 200, 255)    # highlight colour


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)

    vals = depth[finite]
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        hi = lo + 1e-3

    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    norm[~finite] = 0.0
    gray = (255.0 * (1.0 - norm)).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _semantic_to_rgb(semantic: np.ndarray) -> np.ndarray:
    sem = np.asarray(semantic, dtype=np.int64)
    rgb = np.zeros((*sem.shape[:2], 3), dtype=np.uint8)
    mask = sem >= 0
    vals = sem[mask]
    rgb[..., 0][mask] = ((vals * 37) % 255).astype(np.uint8)
    rgb[..., 1][mask] = ((vals * 17 + 73) % 255).astype(np.uint8)
    rgb[..., 2][mask] = ((vals * 29 + 151) % 255).astype(np.uint8)
    return rgb


def _collect_semantic_legend(
    sem_id: np.ndarray,
    sem_name: np.ndarray,
    *,
    min_area_frac: float = 0.005,
) -> list[tuple]:
    """Return [(name, (r, g, b), (cx_frac, cy_frac)), ...] sorted by area desc.

    Colors are derived from the same hash used in ``_semantic_to_rgb`` so the
    legend swatches match the colors drawn on the semantic image. The centroid
    is returned as fractions of the raw image size so the caller can scale it
    to the displayed panel dimensions.
    """
    sem_id = np.asarray(sem_id, dtype=np.int64)
    raw_h, raw_w = sem_id.shape[:2]
    total_px = raw_h * raw_w
    min_area = max(1, int(total_px * min_area_frac))

    entries: list[tuple] = []
    for uid in np.unique(sem_id):
        uid_int = int(uid)
        if uid_int < 0:
            continue
        mask = sem_id == uid
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask)
        name = str(sem_name[ys[0], xs[0]])
        if not name:
            continue
        color = (
            (uid_int * 37) % 255,
            (uid_int * 17 + 73) % 255,
            (uid_int * 29 + 151) % 255,
        )
        centroid = (float(xs.mean() / raw_w), float(ys.mean() / raw_h))
        entries.append((name, color, area, centroid))

    entries.sort(key=lambda e: -e[2])
    return [(name, color, centroid) for name, color, _, centroid in entries]


def _mark_semantic_numbers(img, legend: list[tuple]) -> None:
    """Draw circled index markers on the semantic image at each category's centroid.

    ``legend`` is the output of :func:`_collect_semantic_legend`; the marker
    number matches the entry's 1-based index in that list.
    """
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    r = 11  # circle radius
    for idx, (_, color, (cx_frac, cy_frac)) in enumerate(legend, start=1):
        cx = int(cx_frac * img.width)
        cy = int(cy_frac * img.height)
        # use white text on dark fill if the category colour is bright, else
        # dark text on white fill — always contrasting against the tag
        luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        bg   = (20, 20, 25) if luma > 140 else (240, 240, 240)
        fg   = (240, 240, 240) if luma > 140 else (20, 20, 25)
        draw.ellipse(
            [(cx - r, cy - r), (cx + r, cy + r)],
            fill=bg, outline=color, width=3,
        )
        draw.text((cx, cy), str(idx), fill=fg, font=font, anchor="mm")


class EpisodeVisualizer:

    def __init__(self, viz_dir: Path, cfg: dict) -> None:
        self._viz_dir = viz_dir
        self._enabled = cfg.get("enabled", False)
        self._frame_skip = max(1, cfg.get("frame_skip", 1))
        self._info_w = cfg.get("info_width", 300)

        self._episode: Optional[LandmarkRxREpisode] = None
        self._ep_dir: Optional[Path] = None
        self._step: int = 0

    def on_reset(self, episode: LandmarkRxREpisode, obs: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        self._episode = episode
        self._step = 0
        self._ep_dir = self._viz_dir / str(episode.instruction_id)
        self._ep_dir.mkdir(parents=True, exist_ok=True)
        self._save(obs, action=None)

    def on_step(
        self,
        action: int,
        obs: Dict[str, Any],
        done: bool,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._enabled:
            return
        self._step += 1
        if self._step % self._frame_skip != 0 and not done:
            return
        self._save(obs, action=action)

    def on_episode_end(self, metrics: Dict[str, float]) -> None:
        pass  # final frame already written in on_step with done=True

    def _save(self, obs: Dict[str, Any], action: Optional[int]) -> None:
        from PIL import Image
        canvas = _compose(
            obs=obs,
            episode=self._episode,
            step=self._step,
            action=action,
            info_w=self._info_w,
        )
        Image.fromarray(canvas).save(self._ep_dir / f"{self._step:04d}.png")


# ---------------------------------------------------------------------------
#  Frame composition
# ---------------------------------------------------------------------------

def _compose(
    obs: Dict[str, Any],
    episode: Optional[LandmarkRxREpisode],
    step: int,
    action: Optional[int],
    info_w: int,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    # ── RGB-D-S observation from Habitat's rgbds_agent ────────────────
    blank = np.zeros((256, 1024, 3), dtype=np.uint8)
    rgb = obs.get("rgb_viz", obs.get("rgb", blank))
    obs_h, obs_w = rgb.shape[:2]
    obs_img = Image.fromarray(rgb)
    extra_panels = []
    if "depth" in obs:
        depth_img = Image.fromarray(_depth_to_rgb(obs["depth"]))
        depth_img = depth_img.resize((obs_w, obs_h), Image.BILINEAR)
        _draw_label(depth_img, "DEPTH")
        extra_panels.append(depth_img)
    sem_src = obs.get("semantic_id", obs.get("semantic"))
    legend: list[tuple] = []
    if sem_src is not None:
        semantic_img = Image.fromarray(_semantic_to_rgb(sem_src))
        semantic_img = semantic_img.resize((obs_w, obs_h), Image.NEAREST)
        if "semantic_id" in obs and "semantic_name" in obs:
            legend = _collect_semantic_legend(
                obs["semantic_id"], obs["semantic_name"]
            )
            _mark_semantic_numbers(semantic_img, legend)
        _draw_label(semantic_img, "SEMANTIC")
        extra_panels.append(semantic_img)

    obs_panel = Image.new(
        "RGB",
        (obs_w, obs_h + sum(panel.height for panel in extra_panels)),
        color=_BG,
    )
    obs_panel.paste(obs_img, (0, 0))
    y = obs_h
    for panel in extra_panels:
        obs_panel.paste(panel, (0, y))
        y += panel.height

    # ── right panel: info, same height ───────────────────────────────
    panel_h = obs_panel.height
    info_img = Image.new("RGB", (info_w, panel_h), color=_BG)
    draw = ImageDraw.Draw(info_img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    lines: list[tuple] = []   # (text, colour, bold)

    if episode is not None:
        lines += [
            (f"scan     : {episode.scan}", _FG, False),
            (f"instr_id : {episode.instruction_id}", _ACCENT, True),
            ("", _FG, False),
        ]

    action_str = _ACTION_LABELS.get(action, "—") if action is not None else "START"
    lines += [
        (f"step   : {step:04d}", _FG, False),
        (f"action : {action_str}", _ACCENT, True),
        ("", _FG, False),
    ]

    if episode is not None:
        instr_chars = (info_w - 8) // 7   # ~7px per char at font size 12
        wrapped = textwrap.wrap(episode.instruction, width=instr_chars)
        lines += [(ln, (220, 210, 130), False) for ln in wrapped]

    y = 6
    line_h = 16
    for text, colour, bold in lines:
        draw.text((6, y), text, fill=colour, font=font_bold if bold else font)
        y += line_h

    # ── semantic legend: index + colour swatch + MP40 category name ──
    if legend:
        y += line_h // 2
        draw.text((6, y), "CATEGORIES:", fill=_ACCENT, font=font_bold)
        y += line_h
        sw = 12
        for idx, (name, color, _centroid) in enumerate(legend, start=1):
            if y + line_h > panel_h - 4:
                break
            draw.text((6, y), f"{idx:>2}", fill=_FG, font=font_bold)
            sw_x = 6 + 18
            draw.rectangle(
                [(sw_x, y + 2), (sw_x + sw, y + 2 + sw)],
                fill=color, outline=(60, 60, 70),
            )
            draw.text((sw_x + sw + 6, y), name, fill=_FG, font=font)
            y += line_h

    # ── stitch: [360° panorama | info panel] ─────────────────────────
    canvas = Image.new("RGB", (obs_w + info_w, panel_h))
    canvas.paste(obs_panel, (0, 0))
    canvas.paste(info_img, (obs_w, 0))
    return np.array(canvas, dtype=np.uint8)


# ---------------------------------------------------------------------------
#  Visibility checker visualization
# ---------------------------------------------------------------------------

_OBS_BG     = (20,  20,  30)
_OBS_ACCENT = (100, 200, 255)
_OBS_YELLOW = (255, 220, 80)

_STATUS_COLORS = {
    "UNIQUE":      (80,  200, 120),
    "AMBIGUOUS":   (255, 140,  40),
    "NOT VISIBLE": (160, 160, 160),
    "NOT FOUND":   (220,  80,  80),
}


def _obs_load_fonts(size_sm: int = 12, size_lg: int = 15):
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ]
    for path in candidates:
        try:
            return (ImageFont.truetype(path, size_sm),
                    ImageFont.truetype(path, size_lg))
        except Exception:
            pass
    fallback = ImageFont.load_default()
    return fallback, fallback


def _obs_label_bar(img, text: str) -> None:
    from PIL import ImageDraw
    font_sm, _ = _obs_load_fonts()
    draw  = ImageDraw.Draw(img)
    bar_h = 20
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=(30, 30, 50))
    draw.text((5, 4), text, fill=_OBS_ACCENT, font=font_sm)


def _uniqueness_status(result: Dict) -> str:
    u = result.get("unique")
    if u is None:
        return "NOT FOUND"
    if result.get("visible_count", 0) == 0:
        return "NOT VISIBLE"
    return "UNIQUE" if u else "AMBIGUOUS"


def _obs_info_panel(width: int, height: int,
                    episode_id: int, scan: str,
                    sub_idx: int, sub_total: int,
                    landmark: str, instruction: str,
                    result: Optional[Dict] = None,
                    original: str = "") -> "Any":
    from PIL import Image, ImageDraw
    font_sm, font_lg = _obs_load_fonts()

    panel = Image.new("RGB", (width, height), color=_OBS_BG)
    draw  = ImageDraw.Draw(panel)
    y, line_h = 10, 18
    chars = max(1, (width - 12) // 7)

    def row(text, colour=(255, 255, 255), font=None):
        nonlocal y
        draw.text((8, y), text, fill=colour, font=font or font_sm)
        y += line_h

    def sep():
        nonlocal y
        draw.line([(8, y + 4), (width - 8, y + 4)], fill=(60, 60, 80), width=1)
        y += 14

    row(f"ep {episode_id}", _OBS_ACCENT, font_lg)
    row(f"scan: {scan}", colour=(160, 160, 180))
    row(f"sub-path {sub_idx + 1} / {sub_total}")
    sep()

    if result is not None:
        status     = _uniqueness_status(result)
        status_col = _STATUS_COLORS[status]
        bar_h = 26
        draw.rectangle([(0, y - 2), (width, y + bar_h)], fill=status_col)
        draw.text((8, y + 4), status, fill=(255, 255, 255), font=font_lg)
        y += bar_h + 6
        vis   = result.get("visible_count", 0)
        total = result.get("total_in_scene", 0)
        cat   = result.get("matched_category") or "—"
        row(f"visible: {vis} / {total} in scene", colour=status_col)
        row(f"category: {cat}", colour=(180, 180, 180))
        sep()

    row("LANDMARK:", _OBS_YELLOW)
    for line in textwrap.wrap(landmark or "(none)", chars):
        row(f"  {line}", _OBS_YELLOW)
    sep()
    row("landmark instruction:", colour=(160, 160, 180))
    for line in textwrap.wrap(instruction or "(none)", chars):
        row(f"  {line}", colour=(200, 200, 200))
    sep()
    if original:
        row("original:", colour=(160, 160, 180))
        for line in textwrap.wrap(original, chars):
            row(f"  {line}", colour=(180, 200, 180))
        sep()
    row("360° panorama", colour=(120, 120, 140))

    return panel


# ---------------------------------------------------------------------------
#  Sub-path visibility visualization
# ---------------------------------------------------------------------------

def heading_toward(pos_from: np.ndarray, pos_to: np.ndarray) -> float:
    """Clockwise heading from north (−Z) in radians, to face pos_to from pos_from."""
    dx = float(pos_to[0] - pos_from[0])
    dz = float(pos_to[2] - pos_from[2])
    return math.atan2(dx, -dz)


def _draw_label(img, text: str) -> None:
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    bar_h = 18
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=(30, 30, 40))
    draw.text((4, 3), text, fill=_VIS_ACCENT, font=font)


def _make_vis_info_panel(
    width: int,
    height: int,
    result: Dict,
    sub_idx: int,
    sub_total: int,
    sub_instruction: str,
    episode_id: int,
    scan: str,
) -> "Any":
    from PIL import Image, ImageDraw, ImageFont
    panel = Image.new("RGB", (width, height), color=_VIS_BG)
    draw  = ImageDraw.Draw(panel)
    try:
        font      = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 11)
        font_lg   = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
    except Exception:
        font = font_bold = font_lg = ImageFont.load_default()

    visible     = result.get("visible", True)
    status_col  = _VIS_VISIBLE if visible else _VIS_BLOCKED
    status_text = "VISIBLE" if visible else "BLOCKED"

    draw.rectangle([(0, 0), (width, 28)], fill=status_col)
    draw.text((6, 6), status_text, fill=(255, 255, 255), font=font_lg)

    y, line_h = 36, 16
    chars = (width - 8) // 7

    def row(text, colour=_VIS_FG, bold=False):
        nonlocal y
        draw.text((6, y), text, fill=colour, font=font_bold if bold else font)
        y += line_h

    def blank():
        nonlocal y
        y += line_h // 2

    row(f"episode  : {episode_id}", _VIS_ACCENT, bold=True)
    row(f"scan     : {scan}")
    row(f"sub-path : {sub_idx + 1} / {sub_total}")
    blank()
    dist = result.get("distance", 0.0)
    row(f"distance : {dist:.2f} m")
    if not visible:
        obs = result["obstacle"]
        row(f"hit at   : {obs['hit_distance']:.2f} m  ({obs['hit_fraction']*100:.0f}%)",
            _VIS_BLOCKED, bold=True)
        row(f"obstacle : {obs['semantic_cat']}", _VIS_BLOCKED, bold=True)
    blank()
    if sub_instruction:
        row("instruction:", _VIS_ACCENT)
        for line in textwrap.wrap(sub_instruction, width=chars):
            row(line, colour=(220, 210, 130))

    return panel


def save_subpath_viz(
    checker: "VisibilityChecker",
    pos_start: np.ndarray,
    pos_end: np.ndarray,
    result: Dict,
    out_path: Path,
    sub_idx: int = 0,
    sub_total: int = 1,
    sub_instruction: str = "",
    episode_id: int = 0,
    scan: str = "",
) -> None:
    """Render and save a 3-panel PNG for one sub-path visibility check.

    Layout: [start→end view | end→start view | info panel]
    """
    from PIL import Image

    h_s2e = heading_toward(pos_start, pos_end)
    h_e2s = heading_toward(pos_end,   pos_start)
    rgb_s = checker.render_rgb(pos_start, h_s2e)
    rgb_e = checker.render_rgb(pos_end,   h_e2s)

    img_h, img_w = rgb_s.shape[:2]
    info_w = max(280, img_w)

    left = Image.fromarray(rgb_s)
    _draw_label(left, "START  →  END")
    mid = Image.fromarray(rgb_e)
    _draw_label(mid, "END  →  START  (back-trace)")

    info = _make_vis_info_panel(info_w, img_h, result,
                                sub_idx, sub_total, sub_instruction,
                                episode_id, scan)

    canvas = Image.new("RGB", (img_w * 2 + info_w, img_h), color=_VIS_BG)
    canvas.paste(left, (0, 0))
    canvas.paste(mid,  (img_w, 0))
    canvas.paste(info, (img_w * 2, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)




def save_obs_strip(
    checker: "VisibilityChecker",
    pos_end: np.ndarray,
    heading: float,
    episode_id: int,
    scan: str,
    sub_idx: int,
    sub_total: int,
    landmark: str,
    instruction: str,
    out_path: Path,
    result: Optional[Dict] = None,
    original: str = "",
    img_w: int = 320,
    img_h: int = 240,
) -> None:
    """Render RGB-D observations from the shared rendering agent.

    Layout:
        info panel (left, full height) | RGB panorama
                                       | Depth image, when available
    """
    from PIL import Image

    info_w   = 220
    heading  = heading % (2 * math.pi)
    visible_ids = (result or {}).get("visible_ids", [])

    obs      = checker.render_observation(pos_end, heading)
    rgb      = obs.get("rgb_viz", obs.get("rgb"))
    pano_w   = round(img_h * rgb.shape[1] / rgb.shape[0])
    vfov_deg = rgb.shape[0] / rgb.shape[1] * 360.0

    view = Image.fromarray(rgb).resize((pano_w, img_h), Image.LANCZOS)
    _obs_label_bar(view, "360° equirectangular")
    rows = [view]

    if "depth" in obs:
        depth = Image.fromarray(_depth_to_rgb(obs["depth"]))
        depth = depth.resize((pano_w, img_h), Image.BILINEAR)
        _obs_label_bar(depth, "depth")
        rows.append(depth)

    total_h = img_h * len(rows)
    canvas = Image.new("RGB", (info_w + pano_w, total_h), color=_OBS_BG)
    canvas.paste(
        _obs_info_panel(info_w, total_h, episode_id, scan,
                        sub_idx, sub_total, landmark, instruction,
                        result=result, original=original),
        (0, 0),
    )
    for row_idx, row_img in enumerate(rows):
        canvas.paste(row_img, (info_w, row_idx * img_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
