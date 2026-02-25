"""
Tests for the LLM Robot Control pipeline.
Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_models import (
    ActionPlan,
    ActionType,
    BoundingBox,
    RobotAction,
    SceneGraph,
    SceneObject,
    SpatialRelation,
)
from perception.scene_perceiver import ScenePerceiver
from planner.llm_planner import LLMPlanner
from executor.action_executor import ActionExecutor
from simulation.sim_environment import SimulationEnvironment


# =============================================================================
# Scene / Data Model Tests
# =============================================================================

class TestDataModels:
    def test_scene_object_str(self):
        obj = SceneObject("block", "manipulable", (0.5, 0.5),
                          color="red", size="small")
        assert "red" in str(obj)
        assert "block" in str(obj)

    def test_scene_graph_to_text(self):
        scene = SceneGraph(
            objects=[
                SceneObject("red block", "manipulable", (0.3, 0.5)),
                SceneObject("table", "furniture", (0.5, 0.8)),
            ],
            relations=[
                SpatialRelation("red block", "on", "table"),
            ],
        )
        text = scene.to_text()
        assert "red block" in text
        assert "table" in text
        assert "on" in text

    def test_scene_graph_find_object(self):
        scene = SceneGraph(
            objects=[
                SceneObject("red block", "manipulable", (0.3, 0.5), color="red"),
                SceneObject("blue cup", "container", (0.7, 0.4), color="blue"),
            ],
        )
        found = scene.get_object_by_name("cup")
        assert found is not None
        assert found.name == "blue cup"

        not_found = scene.get_object_by_name("banana")
        assert not_found is None

    def test_bounding_box_center(self):
        bbox = BoundingBox(0, 0, 10, 10)
        assert bbox.center == (5.0, 5.0)

        bbox_3d = BoundingBox(0, 0, 10, 10, 0, 6)
        assert bbox_3d.center == (5.0, 5.0, 3.0)

    def test_action_plan_iteration(self):
        plan = ActionPlan(
            task="test",
            actions=[
                RobotAction(ActionType.MOVE_TO, position=(0.5, 0.5, 0.1)),
                RobotAction(ActionType.GRASP, target="block"),
            ],
        )
        assert len(plan) == 2
        actions = list(plan)
        assert actions[0].action_type == ActionType.MOVE_TO


# =============================================================================
# Perception Tests
# =============================================================================

class TestPerception:
    def test_mock_scene(self):
        perceiver = ScenePerceiver()
        scene = perceiver._mock_scene()
        assert len(scene.objects) > 0
        assert len(scene.relations) > 0
        assert scene.to_text() != ""

    def test_color_extraction(self):
        perceiver = ScenePerceiver()
        assert perceiver._extract_color("red block") == "red"
        assert perceiver._extract_color("big table") is None

    def test_category_inference(self):
        perceiver = ScenePerceiver()
        assert perceiver._infer_category("red block") == "manipulable"
        assert perceiver._infer_category("table") == "furniture"
        assert perceiver._infer_category("bottle") == "container"


# =============================================================================
# Planner Tests
# =============================================================================

class TestPlanner:
    def _make_scene(self) -> SceneGraph:
        return SceneGraph(
            objects=[
                SceneObject("red block", "manipulable", (0.3, 0.5), color="red"),
                SceneObject("blue block", "manipulable", (0.5, 0.5), color="blue"),
                SceneObject("table", "furniture", (0.5, 0.8)),
                SceneObject("green cup", "container", (0.7, 0.4), color="green"),
            ],
            relations=[
                SpatialRelation("red block", "on", "table"),
                SpatialRelation("blue block", "on", "table"),
            ],
        )

    def test_mock_plan_pick_place(self):
        planner = LLMPlanner()
        scene = self._make_scene()
        plan = planner.plan_mock(
            "pick up the red block and place it on the table", scene
        )
        assert plan.success
        assert len(plan.actions) > 0
        # Should contain grasp and release
        action_types = [a.action_type for a in plan.actions]
        assert ActionType.GRASP in action_types
        assert ActionType.RELEASE in action_types

    def test_mock_plan_push(self):
        planner = LLMPlanner()
        scene = self._make_scene()
        plan = planner.plan_mock("push the blue block", scene)
        assert plan.success
        action_types = [a.action_type for a in plan.actions]
        assert ActionType.PUSH in action_types

    def test_find_referenced_object(self):
        planner = LLMPlanner()
        scene = self._make_scene()
        obj = planner._find_referenced_object(
            "pick up the red block", scene, hint="pick"
        )
        assert obj is not None
        assert "red" in obj.name.lower()

    def test_llm_response_parsing(self):
        """Test that we can parse a mock LLM JSON response."""
        planner = LLMPlanner()
        scene = self._make_scene()

        mock_response = json.dumps({
            "reasoning": "Test reasoning",
            "actions": [
                {
                    "action": "move_to",
                    "params": {"target": "red block", "position": [0.3, 0.5, 0.15]},
                    "description": "Move above red block",
                },
                {
                    "action": "grasp",
                    "params": {"target": "red block"},
                    "description": "Grasp the red block",
                },
            ],
            "feasible": True,
            "error": None,
        })

        plan = planner._parse_response(mock_response, "test task", scene)
        assert plan.success
        assert len(plan.actions) == 2
        assert plan.actions[0].action_type == ActionType.MOVE_TO
        assert plan.actions[1].action_type == ActionType.GRASP


# =============================================================================
# Executor Tests
# =============================================================================

class TestExecutor:
    def test_dry_run_execution(self):
        plan = ActionPlan(
            task="test",
            actions=[
                RobotAction(ActionType.MOVE_TO, position=(0.3, 0.5, 0.1),
                            description="Move to position"),
                RobotAction(ActionType.GRASP, target="block",
                            description="Grasp block"),
                RobotAction(ActionType.MOVE_TO, position=(0.7, 0.5, 0.1),
                            description="Move to target"),
                RobotAction(ActionType.RELEASE, target="block",
                            description="Release block"),
            ],
        )
        executor = ActionExecutor(dry_run=True)
        result = executor.execute(plan)
        assert result.success
        assert len(result.steps) == 4

    def test_bounds_check(self):
        plan = ActionPlan(
            task="test",
            actions=[
                RobotAction(ActionType.MOVE_TO, position=(999.0, 999.0, 999.0)),
            ],
        )
        executor = ActionExecutor(dry_run=True)
        result = executor.execute(plan)
        assert not result.success  # Should fail bounds check

    def test_double_grasp_fails(self):
        plan = ActionPlan(
            task="test",
            actions=[
                RobotAction(ActionType.GRASP, target="a"),
                RobotAction(ActionType.GRASP, target="b"),  # Should fail
            ],
        )
        executor = ActionExecutor(dry_run=True)
        result = executor.execute(plan)
        assert not result.success

    def test_failed_plan_not_executed(self):
        plan = ActionPlan(task="test", success=False, error_message="nope")
        executor = ActionExecutor(dry_run=True)
        result = executor.execute(plan)
        assert not result.success


# =============================================================================
# Simulation Tests
# =============================================================================

class TestSimulation:
    def test_setup_from_scene(self):
        scene = SceneGraph(
            objects=[
                SceneObject("red block", "manipulable", (0.3, 0.5), color="red"),
                SceneObject("table", "furniture", (0.5, 0.8)),
            ],
        )
        sim = SimulationEnvironment(headless=True)
        sim.setup_from_scene(scene)
        assert "red block" in sim.objects
        assert "table" in sim.objects

    def test_headless_execution(self):
        scene = SceneGraph(
            objects=[
                SceneObject("red block", "manipulable", (0.3, 0.5), color="red"),
                SceneObject("table", "furniture", (0.5, 0.8)),
            ],
        )
        plan = ActionPlan(
            task="test",
            actions=[
                RobotAction(ActionType.MOVE_TO, position=(0.3, 0.5, 0.1)),
            ],
        )
        sim = SimulationEnvironment(headless=True)
        sim.setup_from_scene(scene)
        result = sim.execute_plan(plan, animate=False)
        assert result

    def test_export_scene_graph(self):
        scene = SceneGraph(
            objects=[
                SceneObject("red block", "manipulable", (0.3, 0.5), color="red"),
            ],
        )
        sim = SimulationEnvironment(headless=True)
        sim.setup_from_scene(scene)
        exported = sim.get_scene_graph()
        assert len(exported.objects) == 1
