import os
import numpy as np
from typing import Dict, List


"""
SUMO 辅助采集工具（无路径/构建逻辑）：
- get_lane_group_info: 聚合车道组占有率、车辆数、平均速度
- get_tls_state: 读取指定路口当前相位与剩余时间
- get_cav_ratio: 计算当前仿真中的 CAV 渗透率

说明：本文件不再承担 SUMO 路径解析、netconvert 构建或 TraCI 启动职责；
训练/评估脚本与环境类直接依赖系统环境变量 `SUMO_HOME` 与各自的 sumocfg。
"""


def get_lane_group_info(lane_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """Aggregate lane metrics for groups.

    Returns a dict per group: { occupancy, veh_count, mean_speed_mps }.
    Missing lanes are tolerated; empty groups yield zeros.
    """
    import traci
    out: Dict[str, Dict[str, float]] = {}
    for gname, lanes in lane_groups.items():
        occ_vals: List[float] = []
        speeds: List[float] = []
        veh_total = 0
        for lane_id in lanes:
            try:
                occ_vals.append(float(traci.lane.getLastStepOccupancy(lane_id)))
                speeds.append(float(traci.lane.getLastStepMeanSpeed(lane_id)))
                veh_total += int(traci.lane.getLastStepVehicleNumber(lane_id))
            except Exception:
                # tolerate missing lane ids
                continue
        if len(occ_vals) == 0:
            out[gname] = {"occupancy": 0.0, "veh_count": 0.0, "mean_speed_mps": 0.0}
        else:
            out[gname] = {
                "occupancy": float(np.mean(occ_vals)),
                "veh_count": float(veh_total),
                "mean_speed_mps": float(np.mean(speeds)) if len(speeds) > 0 else 0.0,
            }
    return out


def get_tls_state(tls_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """Return current phase index and remaining time for each TLS id.

    Format: { tls_id: { phase_index: int, remaining_s: float } }
    """
    import traci
    t = float(traci.simulation.getTime())
    res: Dict[str, Dict[str, float]] = {}
    for tid in tls_ids:
        try:
            phase_idx = int(traci.trafficlight.getPhase(tid))
            next_sw = float(traci.trafficlight.getNextSwitch(tid))
            remaining = max(0.0, next_sw - t)
            res[tid] = {"phase_index": phase_idx, "remaining_s": remaining}
        except Exception:
            res[tid] = {"phase_index": -1, "remaining_s": 0.0}
    return res


def get_cav_ratio(cav_type_id: str = "CAV") -> float:
    """Compute CAV penetration ratio among vehicles currently in simulation."""
    import traci
    try:
        vids = traci.vehicle.getIDList()
    except Exception:
        return 0.0
    if len(vids) == 0:
        return 0.0
    cav = 0
    for vid in vids:
        try:
            if traci.vehicle.getTypeID(vid) == cav_type_id:
                cav += 1
        except Exception:
            continue
    return float(cav) / float(len(vids))


__all__ = [
    "get_lane_group_info",
    "get_tls_state",
    "get_cav_ratio",
]