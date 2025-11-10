import os
from typing import Optional


def get_route_files_value(sumo_cfg_path: str) -> str:
    """Parse the `value` of `<route-files .../>` from a SUMO .sumocfg file.

    Returns the relative path string referenced by the `route-files` tag.
    Raises RuntimeError if the tag cannot be found.
    """
    with open(sumo_cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            if "<route-files" in line and "value=" in line:
                q1 = line.find('"')
                q2 = line.find('"', q1 + 1)
                if q1 != -1 and q2 != -1:
                    return line[q1 + 1 : q2]
    raise RuntimeError("Cannot find <route-files> in SUMO config")


def patch_sumocfg_route(sumo_cfg_path: str, route_rel_value: str) -> None:
    """Replace the `value` of `<route-files .../>` in a SUMO .sumocfg file."""
    with open(sumo_cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    replaced = False
    for line in lines:
        if "<route-files" in line and "value=" in line:
            start = line.find("value=")
            if start != -1:
                qs = line.find('"', start)
                qe = line.find('"', qs + 1)
                if qs != -1 and qe != -1:
                    # Keep any trailing content on the same line (e.g., closing tag and attributes)
                    new_lines.append(line[:qs] + f'"{route_rel_value}"' + line[qe + 1 :])
                    replaced = True
                    continue
        new_lines.append(line)

    if not replaced:
        raise RuntimeError("Failed to patch route-files in SUMO config: tag not found")

    with open(sumo_cfg_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"[sumocfg] patched {sumo_cfg_path} -> route-files={route_rel_value}")


__all__ = [
    "get_route_files_value",
    "patch_sumocfg_route",
]