import os
import yaml
from typing import Any, List, Optional


def load_yaml_bom_safe(path: str) -> dict:
    """Load YAML with BOM-safe decoding and control-char cleanup.

    - Detect common BOMs and decode accordingly.
    - Fallback to several encodings if needed.
    - Strip invisible control chars (BOM/NUL) before parsing.
    """
    with open(path, "rb") as fb:
        data = fb.read()
    if data.startswith(b"\xef\xbb\xbf"):
        enc = "utf-8-sig"
    elif data.startswith(b"\xff\xfe"):
        enc = "utf-16-le"
    elif data.startswith(b"\xfe\xff"):
        enc = "utf-16-be"
    else:
        enc = "utf-8"
    text = ""
    try:
        text = data.decode(enc)
    except Exception:
        decoded = False
        for fallback in ("utf-8", "gb18030", "latin1"):
            try:
                text = data.decode(fallback)
                decoded = True
                break
            except Exception:
                continue
        if not decoded:
            raise RuntimeError(f"Failed to decode {path} with any encoding")
    text = text.replace("\ufeff", "").replace("\x00", "")
    parsed = yaml.safe_load(text)
    return parsed or {}


def validate_env_config(cfg: dict, root_dir: Optional[str] = None) -> None:
    """Validate minimal environment configuration.

    Checks required keys, type/range constraints, and common relationships.
    Raises ValueError on hard violations; prints warnings for suggestions.

    Args:
        cfg: Parsed YAML configuration dict.
        root_dir: Project root for resolving relative paths.
    """
    cfg = cfg or {}
    errors: List[str] = []
    warnings: List[str] = []

    def _warn(msg: str):
        warnings.append(msg)

    def _err(msg: str):
        errors.append(msg)

    # 1) sumo_cfg_relpath
    rel = cfg.get("sumo_cfg_relpath")
    if not isinstance(rel, str) or not rel.strip():
        _err("sumo_cfg_relpath 必须是非空字符串")
    else:
        if root_dir:
            abs_path = os.path.join(root_dir, rel)
            if not os.path.exists(abs_path):
                _err(f"SUMO场景文件不存在: {abs_path}")
        else:
            _warn("未提供 root_dir，跳过 sumo_cfg_relpath 文件存在性检查")

    # 2) discrete_speeds_kph
    speeds = cfg.get("discrete_speeds_kph")
    if not isinstance(speeds, list) or len(speeds) < 3:
        _err("discrete_speeds_kph 必须是长度≥3的列表")
    else:
        try:
            vals = [int(round(float(v))) for v in speeds]
        except Exception:
            _err("discrete_speeds_kph 列表元素必须为数值（可转换为整数）")
            vals = []
        if vals:
            if any(v < 20 or v > 120 for v in vals):
                _err("discrete_speeds_kph 取值需在 [20, 120] km/h 范围内")
            if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
                _err("discrete_speeds_kph 必须严格单调递增")

    # 3) decision_interval
    di = cfg.get("decision_interval")
    try:
        di_int = int(di)
    except Exception:
        di_int = -1
    if di_int <= 0:
        _err("decision_interval 必须为正整数（秒）")
    elif di_int < 2 or di_int > 120:
        _warn("decision_interval 建议在 [5, 60] 秒范围内，以匹配信号周期")

    # 4) max_sim_seconds
    ms = cfg.get("max_sim_seconds")
    try:
        ms_int = int(ms)
    except Exception:
        ms_int = -1
    if ms_int <= 0:
        _err("max_sim_seconds 必须为正整数（秒）")
    elif ms_int < di_int:
        _err("max_sim_seconds 应不小于 decision_interval")

    # 5) lane_groups（必需）
    lg = cfg.get("lane_groups")
    if not isinstance(lg, dict):
        _err("lane_groups 必须为字典，包含 upstream/mid/down 三段")
    else:
        for k in ("upstream", "mid", "down"):
            v = lg.get(k)
            if not isinstance(v, list):
                _err(f"lane_groups.{k} 必须为列表")
            elif len(v) == 0:
                _warn(f"lane_groups.{k} 为空；建议至少配置一个受控车道")

    # 6) lane_groups_wb（可选）
    wblg = cfg.get("lane_groups_wb")
    if wblg is not None:
        if not isinstance(wblg, dict):
            _err("lane_groups_wb 必须为字典（若提供）")
        else:
            for k in ("upstream", "mid", "down"):
                v = wblg.get(k)
                if v is not None and not isinstance(v, list):
                    _err(f"lane_groups_wb.{k} 必须为列表（若提供）")

    # 7) tls_ids
    tls = cfg.get("tls_ids")
    if not isinstance(tls, list) or len(tls) == 0 or not all(isinstance(x, str) and x for x in tls):
        _err("tls_ids 必须为非空字符串列表")

    # 8) penetration
    pen = cfg.get("penetration")
    try:
        pen_f = float(pen)
    except Exception:
        pen_f = -1.0
    if pen_f < 0.0 or pen_f > 1.0:
        _err("penetration 必须在 [0.0, 1.0] 范围内")

    # 9) reward weights（扩展多目标）
    for key in (
        "reward_speed_weight",
        "reward_congestion_weight",
        "reward_delay_weight",
        "reward_queue_overflow_weight",
        "reward_stops_weight",
        "reward_fuel_weight",
        "reward_smoothness_weight",
        "reward_coordination_weight",
        "reward_throughput_weight",
        "reward_demand_robust_weight",
        "reward_change_penalty",
    ):
        val = cfg.get(key)
        try:
            vf = float(val)
        except Exception:
            vf = -1.0
        if vf < 0.0:
            _err(f"{key} 必须为非负数")

    # 10) 可选奖励阈值与裁剪项（若提供则做基本校验）
    for key in (
        "reward_queue_overflow_threshold",
        "reward_delay_max_ratio_extra",
        "reward_clip_min",
        "reward_clip_max",
    ):
        if key in cfg and cfg.get(key) is not None:
            try:
                float(cfg.get(key))
            except Exception:
                _warn(f"{key} 非数值，将使用默认值或忽略")

    # 可选项与弃用项提示
    if "max_steps" in cfg:
        _warn("检测到弃用字段 max_steps：请使用 max_sim_seconds + decision_interval")
    if "action_speeds_mps" in cfg and not speeds:
        _warn("action_speeds_mps 已弃用：建议改用 discrete_speeds_kph（km/h）")

    # gamma 建议区间（如配置了）
    if "gamma" in cfg:
        try:
            g = float(cfg["gamma"])  # type: ignore
            if not (0.90 <= g <= 0.9999):
                _warn("gamma 建议在 [0.90, 0.9999]，以覆盖多个信号周期的长期回报")
        except Exception:
            _warn("gamma 非数值，将由脚本默认值接管")

    # 输出警告，最后处理错误
    for w in warnings:
        print(f"[config][warn] {w}")
    if errors:
        raise ValueError("配置不合法:\n" + "\n".join(f"- {e}" for e in errors))


__all__ = ["load_yaml_bom_safe", "validate_env_config"]