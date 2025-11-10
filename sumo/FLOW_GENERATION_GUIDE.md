# grid1x3 Corridor Flow Generation Guide

This document explains how to rebuild the traffic demand files for the
`grid1x3.net.xml` network from scratch. The steps are intentionally explicit so
that you can reproduce the entire pipeline after deleting previously generated
route files.

## 1. Inspect the network and lane functions

1. Open `sumo/grid1x3.net.xml` in **NETEDIT** or a plain-text editor.
2. Confirm the lane ordering for the main corridor:
   - Edge `W_J0` (west entrance) and `E_J2` (east entrance) each have three
     lanes: index `0` is the right-turn lane, `1` is the through lane, and `2`
     is the left-turn lane. The through lane (`index=1`) is the one controlled
     by the VSL agent.
3. Note the side-street lanes: edges such as `N0_J0`, `S0_J0`, `N1_J1`, etc.
   are two-lane roads with index `0` (right turn) and index `1` (through/left).
4. Write down the movement each lane should serve; we will use these
   associations when defining the demand.

## 2. Define the list of movements and routes

1. For each movement you want to allow, list the exact edge sequence. Example:
   - West straight-through: `W_J0 J0_J1 J1_J2 J2_E`.
   - West to south right-turn: `W_J0 J0_S0`.
   - West to north left-turn: `W_J0 J0_N0`.
2. Repeat the process for the east entrance and for all north/south legs that
   feed the corridor. Keep the movement list in a text file for reference.
3. Decide which movements should contain CAVs. In this project only the
   straight-through flows on the main corridor are controlled by the VSL, so we
   mark only those routes as “eligible for CAV”.

## 3. Choose baseline demand levels

1. Pick an analysis horizon. We assume one hour (3 600 s) because it aligns with
   the configuration in `grid1x3.sumocfg`.
2. Assign a vehicles-per-hour (vph) rate to each movement. A balanced starting
   point is:
   - 900 vph for each mainline straight-through flow.
   - 180 vph for mainline left- and right-turns.
   - 240 vph for each side street movement.
3. If you want heavier or lighter traffic, multiply the straight-through values
   by a common factor (keep the ratio of straight vs. turn movements intact to
   preserve lane discipline).

## 4. Configure vehicle types

1. Define a **human-driven** type with moderate car-following parameters, e.g.
   `accel=2.6`, `decel=4.5`, `sigma=0.5`, `minGap=2.5`.
2. Define a **CAV** type with tighter spacing, e.g. `accel=3.0`, `decel=5.0`,
   `sigma=0.1`, `minGap=1.0`.
3. Ensure both types share the same maximum speed so the only difference is in
   behavior.

## 5. Generate the baseline `.rou.xml`

1. Run the helper script that accompanies this repository:

   ```bash
   python scripts/generate_structured_routes.py \
       --output sumo/grid1x3_structured.rou.xml
   ```

   This writes an all-human demand file using the movement and demand table
   from sections 2–3.
2. Open the generated file and verify it contains:
   - Two `<vType>` definitions (`human` and `cav`).
   - A `<route>` element for each movement.
   - A `<flow>` element with `departLane`, `departSpeed="max"`, and
     `departPos="base"` locking vehicles onto the intended lanes.
3. Update `sumo/grid1x3.sumocfg` so that the `<route-files>` entry points to the
   newly created file.

## 6. Create a CAV penetration variant

1. Decide on the target penetration (e.g., 30 % CAV on the mainline through
   lane).
2. Generate the penetrated demand file using the same script:

   ```bash
   python scripts/generate_structured_routes.py \
       --output sumo/grid1x3_structured.rou.xml \
       --penetration 0.3 \
       --penetration-output sumo/grid1x3_structured.pen30.rou.xml
   ```

   The baseline file is regenerated (all human), and the additional penetration
   file splits the mainline straight flows into human and CAV components.
3. To apply the penetration to every movement (not just the mainline), add the
   flag `--split-all-flows`.

## 7. Validate the demand set

1. Launch a short SUMO simulation using the updated configuration:

   ```bash
   sumo-gui -c sumo/grid1x3.sumocfg
   ```

2. Watch the first few cycles and confirm that:
   - Straight-through vehicles spawn in lane 1 and stay in the controlled lane
     until they leave the corridor.
   - Right-turn vehicles remain in lane 0 and turn without blocking the through
     lane.
   - Side-street vehicles queue in their dedicated lanes and do not overflow
     onto the mainline.
3. If any lane looks congested or underutilized, adjust the `--mainline-scale`
   or `--side-scale` options when regenerating the files and re-run the check.

## 8. Automate batch penetrations (optional)

1. For experiments across multiple penetration levels, run the script in a loop:

   ```bash
   for p in 0.0 0.1 0.3 0.5; do
       python scripts/generate_structured_routes.py \
           --output sumo/grid1x3_structured.rou.xml \
           --penetration "$p" \
           --penetration-output "sumo/grid1x3_structured.pen${p/./}.rou.xml"
   done
   ```

2. Keep the baseline file under version control and store each penetration file
   alongside it. Update your training or evaluation scripts to select the
   appropriate file per run.

Following these steps guarantees that the mainline through lane is populated by
vehicles that already occupy the correct lane at spawn time, eliminating the
late lane-change issue encountered with purely random demand generation.
