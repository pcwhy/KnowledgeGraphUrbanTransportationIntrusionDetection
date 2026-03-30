from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common.project_paths import ASSET_RESULTS_DIR, PAPER_GENERATED_DIR, ensure_pipeline_directories
from experiments.synthetic.urban_transport_kg_ids import (
    ADJACENT_SEGMENTS,
    KnowledgeGraphDetector,
    SEGMENT_TO_INTERSECTION,
    SEGMENT_TO_RSU,
    UrbanTransportExperiment,
)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def build_snapshot(seed: int = 42, attack_type: str = "phantom_congestion", snapshot_tick: int = 7):
    sim = UrbanTransportExperiment(seed=seed)
    messages, _ = sim.run_episode(attack_type)
    detector = KnowledgeGraphDetector()

    attack_message = None
    for message in messages:
        if message.tick > snapshot_tick:
            break
        detector.observe(message)
        if message.malicious and message.tick == snapshot_tick and message.attack_type == attack_type:
            attack_message = message

    if attack_message is None:
        raise RuntimeError("No attack message found for requested snapshot.")

    target_segment = attack_message.segment or ""
    adjacent_segments = sorted(ADJACENT_SEGMENTS.get(target_segment, set()))
    local_segments = [target_segment] + adjacent_segments[:2]
    local_intersections = sorted({SEGMENT_TO_INTERSECTION[s] for s in local_segments})
    local_rsus = sorted({SEGMENT_TO_RSU[s] for s in local_segments})
    local_controllers = [f"C_{intersection}" for intersection in local_intersections]

    local_vehicles: List[Tuple[str, str]] = []
    for node_id, attrs in sorted(detector.state.graph.nodes(data=True)):
        if attrs.get("kind") != "vehicle":
            continue
        segment = detector.state.vehicle_segment(node_id)
        if segment in local_segments:
            local_vehicles.append((node_id, segment))
    local_vehicles = local_vehicles[:4]

    intersection = SEGMENT_TO_INTERSECTION[target_segment]
    controller_state = detector.state.controller_signal(intersection) or "unknown"
    majority_state = detector._majority_signal(intersection) or "unknown"

    densities = {segment: detector.state.segment_density(segment) for segment in local_segments}

    return {
        "seed": seed,
        "attack_type": attack_type,
        "snapshot_tick": snapshot_tick,
        "attack_message_id": attack_message.message_id,
        "attacker_id": attack_message.sender_id,
        "target_segment": target_segment,
        "local_segments": local_segments,
        "adjacent_segments": adjacent_segments,
        "local_rsus": local_rsus,
        "local_intersections": local_intersections,
        "local_controllers": local_controllers,
        "local_vehicles": local_vehicles,
        "densities": densities,
        "controller_state": controller_state,
        "majority_state": majority_state,
        "claimed_density": attack_message.density,
    }


def render_tikz(snapshot: Dict[str, object]) -> str:
    target_segment = str(snapshot["target_segment"])
    local_segments = list(snapshot["local_segments"])
    local_vehicles = list(snapshot["local_vehicles"])
    intersection = SEGMENT_TO_INTERSECTION[target_segment]
    controller_id = f"C_{intersection}"
    rsu_id = SEGMENT_TO_RSU[target_segment]
    densities: Dict[str, int] = snapshot["densities"]  # type: ignore[assignment]

    seg_positions = {
        local_segments[0]: "(0,0)",
        local_segments[1]: "(-3.8,-1.3)" if len(local_segments) > 1 else "(-3.8,-1.3)",
        local_segments[2]: "(4.4,-1.5)" if len(local_segments) > 2 else "(4.4,-1.5)",
    }

    lines: List[str] = [
        "% Auto-generated from a real simulator snapshot.",
        "% Include packages in the preamble:",
        "% \\usepackage{tikz}",
        "% \\usetikzlibrary{arrows.meta,positioning}",
        "\\begin{tikzpicture}[",
        "  >=Latex,",
        "  node distance=1.2cm and 1.6cm,",
        "  entity/.style={draw, rounded corners=2pt, thick, fill=blue!6, minimum width=2.8cm, minimum height=0.9cm, align=center},",
        "  infra/.style={draw, rounded corners=2pt, thick, fill=teal!8, minimum width=2.8cm, minimum height=0.9cm, align=center},",
        "  attr/.style={draw, rounded corners=2pt, fill=gray!10, minimum width=3.3cm, minimum height=0.8cm, align=center, font=\\footnotesize},",
        "  rel/.style={-Latex, thick},",
        "  rel2/.style={-Latex, semithick, dashed}",
        "]",
        f"\\node[entity] (seg_main) at {seg_positions[local_segments[0]]} {{Segment ({latex_escape(local_segments[0])})}};",
    ]

    if len(local_segments) > 1:
        lines.append(f"\\node[entity] (seg_left) at {seg_positions[local_segments[1]]} {{Segment ({latex_escape(local_segments[1])})}};")
        lines.append("\\draw[rel] (seg_left) -- node[pos=0.58, above, sloped, fill=white, inner sep=1pt, font=\\scriptsize] {adjacent to} (seg_main);")
    if len(local_segments) > 2:
        lines.append(f"\\node[entity] (seg_right) at {seg_positions[local_segments[2]]} {{Segment ({latex_escape(local_segments[2])})}};")
        lines.append("\\draw[rel] (seg_main) -- node[pos=0.55, above, sloped, fill=white, inner sep=1pt, font=\\scriptsize] {adjacent to} (seg_right);")

    lines.extend(
        [
            f"\\node[infra] (rsu) at (0,1.55) {{RSU ({latex_escape(rsu_id)})}};",
            f"\\node[infra] (int) at (0,-2.55) {{Intersection ({latex_escape(intersection)})}};",
            f"\\node[infra] (ctrl) at (4.0,-2.55) {{Controller ({latex_escape(controller_id)})}};",
            f"\\node[attr] (density) at (-4.2,1.55) {{Observed density on {latex_escape(target_segment)}: {densities[target_segment]}}};",
            f"\\node[attr] (claim) at (4.2,1.55) {{Alert claim on {latex_escape(target_segment)}: {snapshot['claimed_density']}}};",
            f"\\node[attr] (ctrlstate) at (4.0,-3.85) {{Signal state: {latex_escape(str(snapshot['controller_state']))}}};",
            f"\\node[attr] (majority) at (0,-3.85) {{Vehicle observations: {latex_escape(str(snapshot['majority_state']))}}};",
            "\\draw[rel] (seg_main) -- node[pos=0.55, right, fill=white, inner sep=1pt, font=\\scriptsize] {monitored by} (rsu);",
            "\\draw[rel] (int) -- node[pos=0.55, above, fill=white, inner sep=1pt, font=\\scriptsize] {controlled by} (ctrl);",
            "\\draw[rel] (seg_main) -- node[pos=0.58, right, fill=white, inner sep=1pt, font=\\scriptsize] {leads to} (int);",
            "\\draw[rel2] (density) -- (rsu);",
            "\\draw[rel2] (claim) -- (rsu);",
            "\\draw[rel2] (ctrlstate) -- (ctrl);",
            "\\draw[rel2] (majority) -- (int);",
        ]
    )

    for index, (vehicle_id, segment) in enumerate(local_vehicles):
        if segment == local_segments[0]:
            coord = "(-5.6,0.45)"
        elif len(local_segments) > 1 and segment == local_segments[1]:
            coord = "(-5.6,-0.65)"
        elif len(local_segments) > 2 and segment == local_segments[2]:
            coord = "(8.2,0.2)"
        else:
            fallback_coords = ["(-5.6,-0.2)", "(-5.6,-1.2)", "(5.6,-0.2)", "(5.6,-1.2)"]
            coord = fallback_coords[index % len(fallback_coords)]
        node_name = f"veh{index}"
        lines.append(f"\\node[entity] ({node_name}) at {coord} {{Vehicle ({latex_escape(vehicle_id)})}};")
        target = "seg_main"
        if len(local_segments) > 1 and segment == local_segments[1]:
            target = "seg_left"
        elif len(local_segments) > 2 and segment == local_segments[2]:
            target = "seg_right"
        if target == "seg_right":
            lines.append(f"\\draw[rel] ({node_name}.south west) -- node[pos=0.2, above, sloped, fill=white, inner sep=1pt, font=\\scriptsize] {{located on}} ({target}.north east);")
            lines.append(f"\\draw[rel2] ({node_name}.south west) to[bend right=30] node[pos=0.46, below, sloped, fill=white, inner sep=1pt, font=\\scriptsize] {{observes}} (int.east);")
        else:
            lines.append(f"\\draw[rel] ({node_name}) -- node[pos=0.52, above, sloped, fill=white, inner sep=1pt, font=\\scriptsize] {{located on}} ({target});")
            lines.append(f"\\draw[rel2] ({node_name}) to[bend right=10] node[pos=0.48, below, sloped, fill=white, inner sep=1pt, font=\\scriptsize] {{observes}} (int);")

    lines.extend(
        [
            "\\end{tikzpicture}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_pipeline_directories()
    PAPER_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot()
    (ASSET_RESULTS_DIR / "kg_snapshot_metadata.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    tikz = render_tikz(snapshot)
    (PAPER_GENERATED_DIR / "kg_snapshot_tikz.tex").write_text(tikz, encoding="utf-8")
    print(f"Wrote {PAPER_GENERATED_DIR / 'kg_snapshot_tikz.tex'}")


if __name__ == "__main__":
    main()
