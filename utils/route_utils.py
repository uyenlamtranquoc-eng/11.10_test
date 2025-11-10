import os
import xml.etree.ElementTree as ET
from typing import Optional, Dict


def get_route_files_value(sumo_cfg_path: str) -> str:
    """Parse `<route-files value="..."/>` from SUMO .sumocfg file."""
    with open(sumo_cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            if "<route-files" in line and "value=" in line:
                q1 = line.find('"')
                q2 = line.find('"', q1 + 1)
                if q1 != -1 and q2 != -1:
                    return line[q1 + 1:q2]
    raise RuntimeError("Cannot find <route-files> in SUMO config")


def patch_sumocfg_route(sumo_cfg_path: str, route_rel_value: str) -> None:
    """Replace `<route-files value>` in a SUMO .sumocfg file."""
    with open(sumo_cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    replaced = False
    for line in lines:
        if "<route-files" in line:
            start = line.find("value=")
            if start != -1:
                prefix = line[:start]
                qs = line.find('"', start)
                qe = line.find('"', qs + 1)
                if qs != -1 and qe != -1:
                    new_lines.append(prefix + f"value=\"{route_rel_value}\"" + line[qe + 1:])
                    replaced = True
                    continue
        new_lines.append(line)
    if not replaced:
        raise RuntimeError("Failed to patch route-files in SUMO config: tag not found")
    with open(sumo_cfg_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"[sumocfg] patched {sumo_cfg_path} -> route-files={route_rel_value}")


def strip_pen_suffix(route_rel: str) -> str:
    """Strip repeated `.penXX` suffixes to get the base .rou filename.

    Example:
    `grid1x3_weighted_mt.rou.pen30.pen45.xml` -> `grid1x3_weighted_mt.rou.xml`
    """
    import re
    name, ext = os.path.splitext(route_rel)
    base_name = re.sub(r"(\.pen\d{2})+$", "", name)
    return base_name + ext


def generate_penetrated_routes_from_base(
    base_route_path: str,
    output_path: str,
    cav_rate: float,
    cfg: Dict,
    rng: Optional[object] = None,
) -> None:
    """Generate a .rou with mixed CAV/HV types from a base .rou preserving departures and routes.

    - Reads CAV/HV parameters from cfg to keep train/eval consistent.
    - Assigns vehicle `type` as CAV or HV according to `cav_rate` using `rng` or Python `random`.
    """
    tree = ET.parse(base_route_path)
    root_in = tree.getroot()

    routes_out = ET.Element("routes")
    vdist = ET.SubElement(routes_out, "vTypeDistribution", id="typedist1")
    # 车辆类型名称：尊重配置中的 CAV/HV 类型 ID
    cav_type_id = str(cfg.get("cav_type_id", "CAV"))
    hv_type_id = str(cfg.get("hv_type_id", "HV"))

    cav_tau = str(cfg.get("cav_tau", 0.5))
    cav_accel = str(cfg.get("cav_accel", 3.0))
    cav_decel = str(cfg.get("cav_decel", 4.5))
    cav_min_gap = str(cfg.get("cav_min_gap", 0.50))
    cav_speed_dev = str(cfg.get("cav_speed_dev", 0.0))
    cav_sigma = str(cfg.get("cav_sigma", 0))

    hv_tau = str(cfg.get("hv_tau", 1.5))
    hv_accel = str(cfg.get("hv_accel", 2.5))
    hv_decel = str(cfg.get("hv_decel", 4.5))
    hv_min_gap = str(cfg.get("hv_min_gap", 0.50))
    hv_speed_dev = str(cfg.get("hv_speed_dev", 0.05))

    ET.SubElement(
        vdist,
        "vType",
        id=cav_type_id,
        length="5.0",
        color="0,0,255",
        minGap=cav_min_gap,
        maxSpeed="27.78",
        carFollowModel="IDM",
        accel=cav_accel,
        decel=cav_decel,
        emergencyDecel="6.5",
        tau=cav_tau,
        speedDev=cav_speed_dev,
        sigma=cav_sigma,
        laneChangeModel="LC2013",
        lcAssertive="1.0",
        lcStrategic="1.0",
        lcCooperative="1.0",
        probability="1.0",
        electricBatteryCapacity="50",
        vehicleClass="electric",
        emissionClass="Energy",
    )

    ET.SubElement(
        vdist,
        "vType",
        id=hv_type_id,
        length="5.0",
        color="255,0,0",
        minGap=hv_min_gap,
        maxSpeed="27.78",
        carFollowModel="IDM",
        accel=hv_accel,
        decel=hv_decel,
        emergencyDecel="6.5",
        tau=hv_tau,
        speedDev=hv_speed_dev,
        laneChangeModel="LC2013",
        lcAssertive="1.0",
        lcStrategic="1.0",
        lcCooperative="1.0",
        probability="0.0",
    )

    import random as pyrand
    rng = rng or pyrand
    for veh in root_in.findall("vehicle"):
        vid = veh.get("id")
        depart = veh.get("depart")
        if vid is None or depart is None:
            continue
        r = veh.find("route")
        edges = r.get("edges") if r is not None else None
        if not edges:
            continue
        vtype = cav_type_id if float(rng.random()) < float(cav_rate) else hv_type_id
        new_v = ET.SubElement(routes_out, "vehicle", id=vid, depart=depart, type=vtype, vClass="passenger")
        ET.SubElement(new_v, "route", edges=edges)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ET.ElementTree(routes_out).write(output_path, xml_declaration=True, encoding="UTF-8")
    print(f"[routes] wrote {output_path} | base={os.path.basename(base_route_path)} | CAV penetration={cav_rate}")


def ensure_route_penetration(route_path: str, expected_rate: float, tolerance: float = 0.02, cav_type_id: str = "CAV") -> bool:
    """Verify the CAV penetration ratio (by configured type id) in a .rou file within tolerance."""
    tree = ET.parse(route_path)
    root = tree.getroot()
    total = 0
    cav = 0
    for veh in root.findall("vehicle"):
        total += 1
        if (veh.get("type") or "") == str(cav_type_id):
            cav += 1
    actual = (float(cav) / float(total)) if total > 0 else 0.0
    ok = abs(actual - float(expected_rate)) <= float(tolerance)
    print(f"[consistency] route {os.path.basename(route_path)} CAV('{cav_type_id}') ratio={actual:.3f} target={expected_rate:.3f} tol=±{tolerance:.3f} => {'OK' if ok else 'MISMATCH'}")
    return ok


__all__ = [
    "get_route_files_value",
    "patch_sumocfg_route",
    "strip_pen_suffix",
    "generate_penetrated_routes_from_base",
    "ensure_route_penetration",
]