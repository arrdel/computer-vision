"""
2D Simulation Environment
--------------------------
A simple pygame-based 2D simulation for testing the LLM planner
without a physical robot. Renders objects, the robot arm, and
animates the execution of action plans.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from models.data_models import (
    ActionPlan,
    ActionType,
    RobotAction,
    SceneGraph,
    SceneObject,
    SpatialRelation,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Simulation World
# =============================================================================

@dataclass
class SimObject:
    """An object in the 2D simulation."""
    name: str
    x: float
    y: float
    width: float = 30.0
    height: float = 30.0
    color: tuple[int, int, int] = (200, 200, 200)
    is_held: bool = False
    category: str = "object"

    def contains(self, px: float, py: float) -> bool:
        return (self.x - self.width / 2 <= px <= self.x + self.width / 2 and
                self.y - self.height / 2 <= py <= self.y + self.height / 2)


@dataclass
class SimRobot:
    """The robot in the 2D simulation."""
    x: float = 400.0
    y: float = 100.0
    gripper_open: bool = True
    held_object: Optional[str] = None
    speed: float = 3.0  # pixels per frame


class SimulationEnvironment:
    """
    2D grid-based simulation for testing robot plans.

    Can run in two modes:
    - **Visual mode** (requires pygame): renders a window with animation.
    - **Headless mode**: runs without display, just updates state.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        headless: bool = False,
    ):
        self.width = width
        self.height = height
        self.headless = headless

        # Simulation state
        self.objects: dict[str, SimObject] = {}
        self.robot = SimRobot(x=width / 2, y=100)
        self.running = False
        self.screen = None
        self.clock = None

        # Colors
        self.COLORS = {
            "red": (220, 60, 60),
            "blue": (60, 60, 220),
            "green": (60, 180, 60),
            "yellow": (220, 220, 60),
            "orange": (220, 140, 40),
            "purple": (140, 60, 200),
            "brown": (139, 90, 43),
            "gray": (160, 160, 160),
            "white": (240, 240, 240),
            "black": (40, 40, 40),
        }

    def setup_from_scene(self, scene: SceneGraph) -> None:
        """Initialize simulation objects from a SceneGraph."""
        self.objects.clear()

        for obj in scene.objects:
            # Map normalized positions to pixel coordinates
            px = obj.position[0] * self.width
            py = obj.position[1] * self.height

            color = self.COLORS.get(obj.color or "", (200, 200, 200))
            size = {"small": 25, "medium": 35, "large": 60}.get(obj.size or "medium", 35)

            # Furniture gets bigger
            if obj.category in ("furniture", "surface"):
                size = max(size, 80)

            self.objects[obj.name] = SimObject(
                name=obj.name,
                x=px, y=py,
                width=size, height=size,
                color=color,
                category=obj.category,
            )

        logger.info(f"Simulation initialized with {len(self.objects)} objects")

    def get_scene_graph(self) -> SceneGraph:
        """Export current simulation state as a SceneGraph."""
        objects = []
        for sim_obj in self.objects.values():
            objects.append(SceneObject(
                name=sim_obj.name,
                category=sim_obj.category,
                position=(sim_obj.x / self.width, sim_obj.y / self.height),
                color=self._reverse_color(sim_obj.color),
            ))

        relations = self._compute_relations()
        scene = SceneGraph(objects=objects, relations=relations)
        scene.description = scene.to_text()
        return scene

    def execute_plan(self, plan: ActionPlan, animate: bool = True) -> bool:
        """
        Execute an action plan in the simulation.

        Returns True if all steps succeeded.
        """
        if not self.headless and animate:
            return self._execute_animated(plan)
        else:
            return self._execute_instant(plan)

    def _execute_instant(self, plan: ActionPlan) -> bool:
        """Execute plan instantly without animation."""
        for action in plan.actions:
            success = self._apply_action(action)
            if not success:
                logger.error(f"Action failed: {action}")
                return False
            logger.info(f"Executed: {action}")
        return True

    def _execute_animated(self, plan: ActionPlan) -> bool:
        """Execute plan with pygame animation."""
        try:
            import pygame
        except ImportError:
            logger.warning("pygame not installed, falling back to headless")
            return self._execute_instant(plan)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("LLM Robot Control Simulation")
        self.clock = pygame.time.Clock()
        self.running = True

        font = pygame.font.SysFont("monospace", 14)
        title_font = pygame.font.SysFont("monospace", 16, bold=True)

        current_step = 0
        action_complete = True

        # Animation state
        target_x, target_y = self.robot.x, self.robot.y

        while self.running and current_step <= len(plan.actions):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
                    break

            if not self.running:
                break

            # Start next action
            if action_complete and current_step < len(plan.actions):
                action = plan.actions[current_step]
                action_complete = False

                if action.action_type == ActionType.MOVE_TO and action.position:
                    target_x = action.position[0] * self.width
                    target_y = action.position[1] * self.height
                elif action.action_type == ActionType.GRASP:
                    self._sim_grasp()
                    action_complete = True
                    current_step += 1
                elif action.action_type == ActionType.RELEASE:
                    self._sim_release()
                    action_complete = True
                    current_step += 1
                else:
                    action_complete = True
                    current_step += 1
            elif action_complete and current_step >= len(plan.actions):
                # Plan complete — keep rendering for a moment
                time.sleep(1)
                break

            # Animate movement
            if not action_complete:
                dx = target_x - self.robot.x
                dy = target_y - self.robot.y
                dist = math.sqrt(dx**2 + dy**2)

                if dist < self.robot.speed:
                    self.robot.x = target_x
                    self.robot.y = target_y
                    self._update_held_object()
                    action_complete = True
                    current_step += 1
                else:
                    self.robot.x += (dx / dist) * self.robot.speed
                    self.robot.y += (dy / dist) * self.robot.speed
                    self._update_held_object()

            # ---- Render ----
            self.screen.fill((30, 30, 40))

            # Draw grid
            for x in range(0, self.width, 40):
                pygame.draw.line(self.screen, (50, 50, 60), (x, 0), (x, self.height))
            for y in range(0, self.height, 40):
                pygame.draw.line(self.screen, (50, 50, 60), (0, y), (self.width, y))

            # Draw objects
            for obj in self.objects.values():
                if obj.is_held:
                    continue
                rect = pygame.Rect(
                    obj.x - obj.width / 2, obj.y - obj.height / 2,
                    obj.width, obj.height,
                )
                pygame.draw.rect(self.screen, obj.color, rect, border_radius=4)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1, border_radius=4)

                # Label
                label = font.render(obj.name, True, (255, 255, 255))
                self.screen.blit(label, (obj.x - label.get_width() / 2, obj.y + obj.height / 2 + 2))

            # Draw robot
            robot_color = (0, 200, 255) if self.robot.gripper_open else (255, 100, 0)
            pygame.draw.circle(self.screen, robot_color,
                               (int(self.robot.x), int(self.robot.y)), 12)
            pygame.draw.circle(self.screen, (255, 255, 255),
                               (int(self.robot.x), int(self.robot.y)), 12, 2)

            # Draw held object attached to robot
            if self.robot.held_object and self.robot.held_object in self.objects:
                held = self.objects[self.robot.held_object]
                rect = pygame.Rect(
                    self.robot.x - held.width / 2,
                    self.robot.y + 14,
                    held.width, held.height,
                )
                pygame.draw.rect(self.screen, held.color, rect, border_radius=4)

            # HUD
            title = title_font.render(f"Task: {plan.task}", True, (255, 255, 200))
            self.screen.blit(title, (10, 10))

            step_text = f"Step {min(current_step + 1, len(plan.actions))}/{len(plan.actions)}"
            if current_step < len(plan.actions):
                step_text += f" — {plan.actions[current_step].description}"
            step_label = font.render(step_text, True, (200, 200, 200))
            self.screen.blit(step_label, (10, 35))

            gripper_text = f"Gripper: {'OPEN' if self.robot.gripper_open else 'CLOSED'}"
            if self.robot.held_object:
                gripper_text += f" (holding: {self.robot.held_object})"
            gripper_label = font.render(gripper_text, True, (180, 180, 180))
            self.screen.blit(gripper_label, (10, self.height - 25))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        return True

    def _apply_action(self, action: RobotAction) -> bool:
        """Apply a single action to the simulation state."""
        if action.action_type == ActionType.MOVE_TO and action.position:
            self.robot.x = action.position[0] * self.width
            self.robot.y = action.position[1] * self.height
            self._update_held_object()
            return True
        elif action.action_type == ActionType.GRASP:
            return self._sim_grasp()
        elif action.action_type == ActionType.RELEASE:
            return self._sim_release()
        return True

    def _sim_grasp(self) -> bool:
        """Grasp the nearest object within reach."""
        self.robot.gripper_open = False
        reach = 40
        for name, obj in self.objects.items():
            dist = math.sqrt((self.robot.x - obj.x)**2 + (self.robot.y - obj.y)**2)
            if dist < reach and obj.category not in ("furniture", "surface"):
                obj.is_held = True
                self.robot.held_object = name
                logger.info(f"Grasped: {name}")
                return True
        logger.warning("Grasp failed: no object within reach")
        return True  # Don't fail the plan, just warn

    def _sim_release(self) -> bool:
        """Release the held object at current position."""
        self.robot.gripper_open = True
        if self.robot.held_object and self.robot.held_object in self.objects:
            obj = self.objects[self.robot.held_object]
            obj.is_held = False
            obj.x = self.robot.x
            obj.y = self.robot.y + 20
            logger.info(f"Released: {self.robot.held_object}")
        self.robot.held_object = None
        return True

    def _update_held_object(self):
        """Move held object with the robot."""
        if self.robot.held_object and self.robot.held_object in self.objects:
            obj = self.objects[self.robot.held_object]
            obj.x = self.robot.x
            obj.y = self.robot.y + 20

    def _compute_relations(self) -> list[SpatialRelation]:
        """Compute spatial relations between simulation objects."""
        relations = []
        objs = list(self.objects.values())
        for i, a in enumerate(objs):
            for j, b in enumerate(objs):
                if i == j:
                    continue
                dx = a.x - b.x
                dy = a.y - b.y
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 100:
                    if abs(dx) > abs(dy):
                        rel = "right_of" if dx > 0 else "left_of"
                    else:
                        rel = "below" if dy > 0 else "above"
                    relations.append(SpatialRelation(a.name, rel, b.name))
        return relations

    def _reverse_color(self, rgb: tuple[int, int, int]) -> Optional[str]:
        """Try to name a color from RGB."""
        for name, c in self.COLORS.items():
            if c == rgb:
                return name
        return None
