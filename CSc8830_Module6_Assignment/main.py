"""
Main entry point for the LLM Robot Control system.

Usage:
    # Simulated environment with mock LLM (no API key needed)
    python main.py --mode sim --task "pick up the red block and place it near the green cup"

    # Simulated environment with real LLM
    python main.py --mode sim --task "pick up the red block" --use-llm

    # Image-based perception with LLM planning
    python main.py --mode image --image scene.jpg --task "grab the cup"

    # Interactive mode
    python main.py --mode interactive
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.data_models import SceneGraph
from perception.scene_perceiver import ScenePerceiver
from planner.llm_planner import LLMPlanner
from executor.action_executor import ActionExecutor
from simulation.sim_environment import SimulationEnvironment


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_simulation(task: str, use_llm: bool = False, animate: bool = True):
    """Run the full pipeline with a simulated environment."""
    print("\n" + "=" * 60)
    print("  LLM Robot Control — Simulation Mode")
    print("=" * 60)

    # 1. Create a simulated scene
    perceiver = ScenePerceiver()
    scene = perceiver._mock_scene()  # Use mock scene for simulation
    print(f"\n📷 Scene perceived with {len(scene.objects)} objects:")
    print(scene.to_text())

    # 2. Plan actions
    planner = LLMPlanner()
    print(f"\n🧠 Planning for task: '{task}'")

    if use_llm:
        plan = planner.plan(task, scene)
    else:
        plan = planner.plan_mock(task, scene)

    print(f"\n📋 Generated plan ({len(plan)} steps):")
    print(plan)
    if plan.reasoning:
        print(f"\n💭 Reasoning: {plan.reasoning}")

    if not plan.success:
        print(f"\n❌ Planning failed: {plan.error_message}")
        return

    # 3. Execute in simulation
    sim = SimulationEnvironment(headless=not animate)
    sim.setup_from_scene(scene)

    print(f"\n🤖 Executing plan...")
    if animate:
        print("   (Close the pygame window or press ESC when done)")
        sim.execute_plan(plan, animate=True)
    else:
        # Also execute with the ActionExecutor for logging
        executor = ActionExecutor(dry_run=True)
        result = executor.execute(plan)
        print(result)


def run_image_mode(image_path: str, task: str, use_llm: bool = True):
    """Run the pipeline with a real image input."""
    print("\n" + "=" * 60)
    print("  LLM Robot Control — Image Mode")
    print("=" * 60)

    # 1. Perceive the scene from the image
    perceiver = ScenePerceiver()
    print(f"\n📷 Perceiving scene from: {image_path}")
    scene = perceiver.perceive(image_path)
    print(scene.to_text())

    # 2. Plan
    planner = LLMPlanner()
    print(f"\n🧠 Planning for task: '{task}'")

    if use_llm:
        plan = planner.plan(task, scene)
    else:
        plan = planner.plan_mock(task, scene)

    print(f"\n📋 Generated plan ({len(plan)} steps):")
    print(plan)

    # 3. Execute (dry run for image mode)
    executor = ActionExecutor(dry_run=True)
    result = executor.execute(plan)
    print(f"\n🤖 Execution result:")
    print(result)


def run_interactive():
    """Interactive mode — enter tasks in a loop."""
    print("\n" + "=" * 60)
    print("  LLM Robot Control — Interactive Mode")
    print("=" * 60)
    print("  Enter tasks in natural language. Type 'quit' to exit.")
    print("  Type 'scene' to see the current scene.")
    print("  Type 'llm' to toggle LLM mode on/off.")
    print("=" * 60)

    perceiver = ScenePerceiver()
    scene = perceiver._mock_scene()
    planner = LLMPlanner()
    executor = ActionExecutor(dry_run=True)
    use_llm = False

    print(f"\n📷 Initial scene:")
    print(scene.to_text())
    print(f"\n🔧 LLM mode: {'ON' if use_llm else 'OFF (using mock planner)'}")

    while True:
        try:
            task = input("\n🗣️  Enter task > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not task:
            continue
        if task.lower() in ("quit", "exit", "q"):
            break
        if task.lower() == "scene":
            print(scene.to_text())
            continue
        if task.lower() == "llm":
            use_llm = not use_llm
            print(f"🔧 LLM mode: {'ON' if use_llm else 'OFF'}")
            continue

        # Plan and execute
        if use_llm:
            plan = planner.plan(task, scene)
        else:
            plan = planner.plan_mock(task, scene)

        print(f"\n📋 Plan ({len(plan)} steps):")
        print(plan)

        result = executor.execute(plan)
        print(f"\n🤖 Result:")
        print(result)

    print("\nGoodbye! 👋")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-Driven Robot Control with Environmental Understanding"
    )
    parser.add_argument(
        "--mode", choices=["sim", "image", "interactive"],
        default="sim",
        help="Execution mode",
    )
    parser.add_argument(
        "--task", type=str,
        default="pick up the red block and place it near the green cup",
        help="High-level task description",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to scene image (for image mode)",
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Use real LLM API (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--no-animate", action="store_true",
        help="Disable pygame animation (headless mode)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.mode == "sim":
        run_simulation(args.task, use_llm=args.use_llm, animate=not args.no_animate)
    elif args.mode == "image":
        if not args.image:
            print("Error: --image path required for image mode")
            sys.exit(1)
        run_image_mode(args.image, args.task, use_llm=args.use_llm)
    elif args.mode == "interactive":
        run_interactive()


if __name__ == "__main__":
    main()
