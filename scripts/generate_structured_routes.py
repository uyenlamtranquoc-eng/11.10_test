from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Iterable, List

from xml.etree import ElementTree as ET


# ----------------------------- data structures -----------------------------


@dataclass(frozen=True)
class FlowSpec:
    """Description of a single SUMO ``<flow>`` element."""

    identifier: str
    route_edges: List[str]
    depart_lane: str
    vehs_per_hour: float
    description: str
    eligible_for_cav: bool
    depart_speed: str = "0"  # queueing-friendly default; override via CLI if needed
    depart_pos: str = "base"


# Default demand table sized for the grid1x3 corridor. Demand values are per
# direction per hour and can be rescaled via command-line arguments.
DEFAULT_FLOW_SPECS: List[FlowSpec] = [
    # --- West entrance at J0 ---
    FlowSpec(
        identifier="west_straight",
        route_edges=["W_J0", "J0_J1", "J1_J2", "J2_E"],
        depart_lane="1",
        vehs_per_hour=400.0,
        description="Mainline straight-through traffic (controlled lane)",
        eligible_for_cav=True,
    ),
    FlowSpec(
        identifier="west_right",
        route_edges=["W_J0", "J0_S0"],
        depart_lane="0",
        vehs_per_hour=400.0,
        description="Westbound to south right-turn movements",
        eligible_for_cav=False,
    ),
    FlowSpec(
        identifier="west_left",
        route_edges=["W_J0", "J0_N0"],
        depart_lane="2",
        vehs_per_hour=400.0,
        description="Westbound to north left-turn movements",
        eligible_for_cav=False,
    ),
    # --- East entrance at J2 ---
    FlowSpec(
        identifier="east_straight",
        route_edges=["E_J2", "J2_J1", "J1_J0", "J0_W"],
        depart_lane="1",
        vehs_per_hour=400.0,
        description="Eastbound straight-through traffic (controlled lane)",
        eligible_for_cav=True,
    ),
    FlowSpec(
        identifier="east_right",
        route_edges=["E_J2", "J2_N2"],
        depart_lane="0",
        vehs_per_hour=400.0,
        description="Eastbound to north right-turn movements",
        eligible_for_cav=False,
    ),
    FlowSpec(
        identifier="east_left",
        route_edges=["E_J2", "J2_S2"],
        depart_lane="2",
        vehs_per_hour=400.0,
        description="Eastbound to south left-turn movements",
        eligible_for_cav=False,
    ),
    # --- Side streets feeding into the corridor ---
    FlowSpec(
        identifier="south0_northbound",
        route_edges=["S0_J0", "J0_J1", "J1_N1"],
        depart_lane="0",
        vehs_per_hour=200.0,
        description="South leg at J0 heading north",
        eligible_for_cav=False,
    ),
    FlowSpec(
        identifier="south1_northbound",
        route_edges=["S1_J1", "J1_J2", "J2_N2"],
        depart_lane="0",
        vehs_per_hour=200.0,
        description="South leg at J1 heading north",
        eligible_for_cav=False,
    ),
    FlowSpec(
        identifier="south2_straight",
        route_edges=["S2_J2", "J2_J1", "J1_J0"],
        depart_lane="1",  # left at J2 -> J2_J1 requires from lane 1
        vehs_per_hour=200.0,
        description="South leg at J2 heading north-west",
        eligible_for_cav=False,
    ),
    FlowSpec(
        identifier="north0_southbound",
        route_edges=["N0_J0", "J0_J1", "J1_S1"],
        depart_lane="1",
        vehs_per_hour=200.0,
        description="North leg at J0 heading south-east",
        eligible_for_cav=False,
    ),
    FlowSpec(
        identifier="north1_southbound",
        route_edges=["N1_J1", "J1_J0", "J0_S0"],
        depart_lane="0",  # right at J1 -> J1_J0 requires from lane 0
        vehs_per_hour=200.0,
        description="North leg at J1 heading south-west",
        eligible_for_cav=False,
    ),
    FlowSpec(
        identifier="north2_straight",
        route_edges=["N2_J2", "J2_J1", "J1_J0"],
        depart_lane="0",  # right at J2 -> J2_J1 requires from lane 0
        vehs_per_hour=200.0,
        description="North leg at J2 heading south-west",
        eligible_for_cav=False,
    ),
]


# ----------------------------- xml utilities ------------------------------


def build_routes_document(
    flow_specs: Iterable[FlowSpec],
    horizon: float,
    cav_penetration: float,
    split_only_mainline: bool,
) -> ET.Element:
    """Create the XML tree for a SUMO routes file."""

    if not 0.0 <= cav_penetration <= 1.0:
        raise ValueError("CAV penetration must be between 0 and 1")

    root = ET.Element(
        "routes",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )

    vtype_human = ET.SubElement(
        root,
        "vType",
        attrib={
            "id": "hdv",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.5",
            "length": "5.0",
            "minGap": "2.5",
            "speedDev": "0.1",
            "tau": "1.0",
            "carFollowModel": "Krauss",
            "color": "255,255,0",
            "maxSpeed": "15",
            "laneChangeModel": "LC2013",
        },
    )
    vtype_human.tail = "\n    "

    vtype_cav = ET.SubElement(
        root,
        "vType",
        attrib={
            "id": "cav",
            "accel": "3.0",
            "decel": "5.0",
            "sigma": "0.1",
            "length": "4.5",
            "minGap": "1.0",
            "speedDev": "0.05",
            "tau": "1.0",
            "carFollowModel": "IDM",
            "color": "255,0,0",
            "maxSpeed": "15",
            "laneChangeModel": "LC2013",
        },
    )
    vtype_cav.tail = "\n    "

    for spec in flow_specs:
        route_id = f"route_{spec.identifier}"
        flow_comment = ET.Comment(spec.description)
        root.append(flow_comment)

        route_element = ET.SubElement(
            root,
            "route",
            attrib={"id": route_id, "edges": " ".join(spec.route_edges)},
        )
        route_element.tail = "\n    "

        human_vph, cav_vph = split_demand(
            spec.vehs_per_hour,
            cav_penetration,
            spec.eligible_for_cav,
            split_only_mainline,
        )

        if human_vph > 0:
            ET.SubElement(
                root,
                "flow",
                attrib=flow_attributes(
                    spec=spec,
                    route_id=route_id,
                    vehs_per_hour=human_vph,
                    vehicle_type="hdv",
                    horizon=horizon,
                ),
            ).tail = "\n    "

        if cav_vph > 0:
            ET.SubElement(
                root,
                "flow",
                attrib=flow_attributes(
                    spec=spec,
                    route_id=route_id,
                    vehs_per_hour=cav_vph,
                    vehicle_type="cav",
                    horizon=horizon,
                ),
            ).tail = "\n    "

    return root


def split_demand(
    vehs_per_hour: float,
    cav_penetration: float,
    eligible_for_cav: bool,
    split_only_mainline: bool,
) -> tuple[float, float]:
    """Split the demand into human and CAV components."""

    if cav_penetration == 0 or (split_only_mainline and not eligible_for_cav):
        return vehs_per_hour, 0.0

    cav_vph = vehs_per_hour * cav_penetration
    human_vph = vehs_per_hour - cav_vph
    return human_vph, cav_vph


def flow_attributes(
    spec: FlowSpec,
    route_id: str,
    vehs_per_hour: float,
    vehicle_type: str,
    horizon: float,
) -> dict[str, str]:
    """Prepare the XML attributes for a flow element."""

    # Prefer using explicit `vehsPerHour` to avoid ambiguity; SUMO expects
    # only one of `number`, `vehsPerHour`, `period`, or `probability`.
    # Keeping it simple and deterministic here.
    return {
        "id": f"flow_{spec.identifier}_{vehicle_type}",
        "type": vehicle_type,
        "route": route_id,
        "begin": "0",
        "end": f"{horizon:g}",
        "departLane": spec.depart_lane,
        "departSpeed": spec.depart_speed,
        "departPos": spec.depart_pos,
        "vehsPerHour": f"{vehs_per_hour:.2f}",
    }


def indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print the XML document in-place."""

    indent_str = "    "
    i = "\n" + level * indent_str
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str
        for child in elem:
            indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i + indent_str
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = "\n" + (level - 1) * indent_str


# ------------------------------- cli parsing -------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to write the baseline all-human route file",
    )
    parser.add_argument(
        "--penetration",
        type=float,
        default=0.0,
        help="Target CAV penetration ratio (0-1) for mainline flows",
    )
    parser.add_argument(
        "--penetration-output",
        type=pathlib.Path,
        help="Optional output path for the penetration-specific route file",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=3600.0,
        help="Simulation horizon in seconds",
    )
    parser.add_argument(
        "--mainline-scale",
        type=float,
        default=1.0,
        help="Scaling factor applied to all mainline straight flows",
    )
    parser.add_argument(
        "--side-scale",
        type=float,
        default=1.0,
        help="Scaling factor applied to side-street demand",
    )
    parser.add_argument(
        "--depart-speed",
        type=str,
        default="0",
        help="Global departSpeed for flows (e.g. '0', 'max', 'desired')",
    )
    parser.add_argument(
        "--depart-pos",
        type=str,
        default="base",
        help="Global departPos for flows (e.g. 'base', 'random', 'free')",
    )
    parser.add_argument(
        "--split-all-flows",
        action="store_true",
        help="If set, the penetration ratio will be applied to every flow",
    )
    return parser.parse_args()


def scale_demand(flow_specs: Iterable[FlowSpec], mainline_scale: float, side_scale: float, depart_speed: str, depart_pos: str) -> List[FlowSpec]:
    scaled_specs: List[FlowSpec] = []
    for spec in flow_specs:
        scale = mainline_scale if spec.eligible_for_cav else side_scale
        scaled_specs.append(
            FlowSpec(
                identifier=spec.identifier,
                route_edges=spec.route_edges,
                depart_lane=spec.depart_lane,
                vehs_per_hour=spec.vehs_per_hour * scale,
                description=spec.description,
                eligible_for_cav=spec.eligible_for_cav,
                depart_speed=depart_speed,
                depart_pos=depart_pos,
            )
        )
    return scaled_specs


def write_routes(path: pathlib.Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indent(root)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()

    scaled_specs = scale_demand(
        DEFAULT_FLOW_SPECS,
        mainline_scale=args.mainline_scale,
        side_scale=args.side_scale,
        depart_speed=args.depart_speed,
        depart_pos=args.depart_pos,
    )

    # Baseline file (all human drivers)
    baseline_root = build_routes_document(
        flow_specs=scaled_specs,
        horizon=args.horizon,
        cav_penetration=0.0,
        split_only_mainline=not args.split_all_flows,
    )
    write_routes(args.output, baseline_root)

    if args.penetration_output:
        penetration_root = build_routes_document(
            flow_specs=scaled_specs,
            horizon=args.horizon,
            cav_penetration=args.penetration,
            split_only_mainline=not args.split_all_flows,
        )
        write_routes(args.penetration_output, penetration_root)


if __name__ == "__main__":
    main()