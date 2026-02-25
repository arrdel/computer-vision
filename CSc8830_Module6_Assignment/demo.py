"""
Demo script — runs the full pipeline with visual output.
Demonstrates: perception → LLM planning → execution in simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.data_models import SceneGraph, SceneObject, SpatialRelation
from perception.scene_perceiver import ScenePerceiver
from planner.llm_planner import LLMPlanner
from executor.action_executor import ActionExecutor
from simulation.sim_environment import SimulationEnvironment


def demo_full_pipeline():
    """
    Run the complete pipeline:
    1. Perceive the environment (mock)
    2. Plan with LLM (mock)
    3. Execute and visualize
    """
    print("=" * 60)
    print("  DEMO: LLM-Driven Robot Control")
    print("=" * 60)

    # --- Step 1: Scene Perception ---
    print("\n[1/3] 📷 Perceiving the environment...")
    perceiver = ScenePerceiver()
    scene = perceiver._mock_scene()

    # You can also create a custom scene:
    custom_scene = SceneGraph(
        objects=[
            SceneObject("red block", "manipulable", (0.25, 0.6),
                        color="red", size="small", confidence=0.95),
            SceneObject("blue block", "manipulable", (0.5, 0.6),
                        color="blue", size="small", confidence=0.90),
            SceneObject("green cup", "container", (0.75, 0.5),
                        color="green", confidence=0.88),
            SceneObject("table", "furniture", (0.5, 0.85),
                        size="large", confidence=0.99),
            SceneObject("yellow platform", "surface", (0.75, 0.75),
                        color="yellow", size="medium", confidence=0.85),
        ],
        relations=[
            SpatialRelation("red block", "on", "table"),
            SpatialRelation("blue block", "on", "table"),
            SpatialRelation("green cup", "on", "table"),
            SpatialRelation("red block", "left_of", "blue block"),
            SpatialRelation("blue block", "left_of", "green cup"),
            SpatialRelation("yellow platform", "on", "table"),
        ],
    )
    custom_scene.description = custom_scene.to_text()

    print(custom_scene.to_text())

    # --- Step 2: LLM Planning ---
    tasks = [
        "pick up the red block and place it on the yellow platform",
        "move the blue block next to the green cup",
        "push the red block to the right",
    ]

    planner = LLMPlanner()

    for task in tasks:
        print(f"\n{'─' * 60}")
        print(f"[2/3] 🧠 Planning: '{task}'")

        plan = planner.plan_mock(task, custom_scene)
        print(f"\n{plan}")

        # --- Step 3: Execute ---
        print(f"\n[3/3] 🤖 Executing (dry run)...")
        executor = ActionExecutor(dry_run=True)
        result = executor.execute(plan)
        print(result)

    # --- Optional: Visual simulation of the first task ---
    print(f"\n{'=' * 60}")
    print("🎮 Launching visual simulation for the first task...")
    print("   Close the window or press ESC when done.")
    print(f"{'=' * 60}")

    plan = planner.plan_mock(tasks[0], custom_scene)
    sim = SimulationEnvironment()
    sim.setup_from_scene(custom_scene)
    sim.execute_plan(plan, animate=True)

    print("\n✅ Demo complete!")


def demo_custom_scene():
    """Quick demo showing how to create and query a custom scene."""
    scene = SceneGraph(
        objects=[
            SceneObject("coffee mug", "container", (0.3, 0.4),
                        color="white", size="small"),
            SceneObject("laptop", "object", (0.6, 0.3),
                        size="large"),
            SceneObject("desk", "furniture", (0.5, 0.7),
                        size="large"),
        ],
        relations=[
            SpatialRelation("coffee mug", "on", "desk"),
            SpatialRelation("laptop", "on", "desk"),
            SpatialRelation("coffee mug", "left_of", "laptop"),
        ],
    )
    scene.description = scene.to_text()

    print("Custom Scene:")
    print(scene.to_text())

    # Find an object
    mug = scene.get_object_by_name("mug")
    if mug:
        print(f"\nFound: {mug.name} at {mug.position}")

    # Plan
    planner = LLMPlanner()
    plan = planner.plan_mock("move the coffee mug to the right side of the desk", scene)
    print(f"\n{plan}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", action="store_true",
                        help="Run the custom scene demo instead")
    args = parser.parse_args()

    if args.custom:
        demo_custom_scene()
    else:
        demo_full_pipeline()
