"""
Shared Habitat-sim RGB-D agent configuration helpers.

The project uses habitat-sim directly, so this module provides the same role
as a Habitat-Lab RGBD(S) agent config without depending on a version-specific
Habitat-Lab import path.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _set_if_present(obj: Any, name: str, value: Any) -> None:
    if value is not None and hasattr(obj, name):
        setattr(obj, name, value)


def _sensor_pose(spec: Any, sensor_height: float) -> Any:
    spec.position = [0.0, sensor_height, 0.0]
    spec.orientation = [0.0, 0.0, 0.0]
    return spec


def make_rgbd_sensor_specs(
    habitat_sim: Any,
    cfg: Dict[str, Any],
    *,
    include_rgb: bool = True,
    include_rgb_viz: bool = True,
    include_depth: bool = True,
) -> List[Any]:
    """Create the RGB, equirectangular RGB visualization, and depth sensors."""
    sensor_h = cfg.get("sensor_height", 1.5)
    rgb_cfg = cfg.get("rgb", {})
    depth_cfg = cfg.get("depth", {})

    specs: List[Any] = []

    if include_rgb:
        rgb_w = rgb_cfg.get("width", 256)
        rgb_h = rgb_cfg.get("height", rgb_w)
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [rgb_h, rgb_w]
        rgb.hfov = rgb_cfg.get("hfov", 90)
        specs.append(_sensor_pose(rgb, sensor_h))

    if include_rgb_viz:
        viz_w = rgb_cfg.get("viz_width", rgb_cfg.get("width", 256) * 4)
        viz_vfov = rgb_cfg.get("vfov", 90)
        viz_h = rgb_cfg.get("viz_height", round(viz_w * viz_vfov / 360.0))
        rgb_viz = habitat_sim.EquirectangularSensorSpec()
        rgb_viz.uuid = "rgb_viz"
        rgb_viz.sensor_type = habitat_sim.SensorType.COLOR
        rgb_viz.resolution = [viz_h, viz_w]
        specs.append(_sensor_pose(rgb_viz, sensor_h))

    if include_depth:
        depth_w = depth_cfg.get("width", rgb_cfg.get("width", 256))
        depth_h = depth_cfg.get("height", depth_w)
        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = [depth_h, depth_w]
        depth.hfov = depth_cfg.get("hfov", rgb_cfg.get("hfov", 90))
        _set_if_present(depth, "min_depth", depth_cfg.get("min_depth"))
        _set_if_present(depth, "max_depth", depth_cfg.get("max_depth"))
        specs.append(_sensor_pose(depth, sensor_h))

    return specs


def make_rgbd_agent_config(
    habitat_sim: Any,
    cfg: Dict[str, Any],
    *,
    include_rgb: bool = True,
    include_rgb_viz: bool = True,
    include_depth: bool = True,
    include_actions: bool = True,
) -> Any:
    """Build a habitat-sim AgentConfiguration with RGB-D sensors."""
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = cfg.get("height", 0.88)
    agent_cfg.radius = cfg.get("radius", 0.18)
    agent_cfg.sensor_specifications = make_rgbd_sensor_specs(
        habitat_sim,
        cfg,
        include_rgb=include_rgb,
        include_rgb_viz=include_rgb_viz,
        include_depth=include_depth,
    )

    if include_actions:
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(cfg.get("forward_step_size", 0.25)),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(cfg.get("turn_angle", 15.0)),
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(cfg.get("turn_angle", 15.0)),
            ),
            "stop": habitat_sim.agent.ActionSpec("stop"),
        }

    return agent_cfg
