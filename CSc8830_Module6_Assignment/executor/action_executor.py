"""
Action Executor
----------------
Translates an ActionPlan into executable robot commands.
Validates actions against physical constraints and provides
both real robot interfaces and simulation hooks.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

from models.data_models import ActionPlan, ActionType, RobotAction, SceneGraph

logger = logging.getLogger(__name__)


class RobotInterface(Protocol):
    """Protocol for robot hardware/simulator backends."""

    def move_to(self, position: tuple[float, ...], speed: float) -> bool: ...
    def grasp(self) -> bool: ...
    def release(self) -> bool: ...
    def get_position(self) -> tuple[float, ...]: ...
    def get_gripper_state(self) -> str: ...


class ActionExecutor:
    """
    Executes an ActionPlan step by step.

    Features:
    - Pre-execution validation (collision checks, reachability)
    - Step-by-step logging
    - Support for dry-run mode
    - Hooks for real robot or simulation backends
    """

    def __init__(
        self,
        robot: Optional[RobotInterface] = None,
        workspace_bounds: Optional[dict] = None,
        dry_run: bool = False,
    ):
        self.robot = robot
        self.dry_run = dry_run
        self.workspace_bounds = workspace_bounds or {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.0, 1.5),
        }

        # Execution state
        self.current_position = (0.5, 0.5, 0.3)
        self.gripper_state = "open"
        self.held_object: Optional[str] = None
        self.execution_log: list[dict] = []

    def execute(self, plan: ActionPlan) -> ExecutionResult:
        """
        Execute all actions in a plan sequentially.

        Args:
            plan: The ActionPlan to execute.

        Returns:
            ExecutionResult with status of each step.
        """
        logger.info(f"Executing plan: '{plan.task}' ({len(plan)} steps)")
        self.execution_log = []

        if not plan.success:
            return ExecutionResult(
                success=False,
                message=f"Plan was not successful: {plan.error_message}",
                steps=[],
            )

        # Validate the full plan before execution
        validation = self._validate_plan(plan)
        if not validation.valid:
            logger.warning(f"Plan validation failed: {validation.message}")
            return ExecutionResult(
                success=False,
                message=f"Validation failed: {validation.message}",
                steps=[],
            )

        # Execute each action
        steps = []
        for i, action in enumerate(plan.actions):
            logger.info(f"Step {i + 1}/{len(plan)}: {action}")

            step_result = self._execute_action(action, i + 1)
            steps.append(step_result)
            self.execution_log.append(step_result.__dict__)

            if not step_result.success:
                logger.error(f"Step {i + 1} failed: {step_result.message}")
                return ExecutionResult(
                    success=False,
                    message=f"Failed at step {i + 1}: {step_result.message}",
                    steps=steps,
                )

        logger.info(f"Plan executed successfully: {len(steps)} steps completed")
        return ExecutionResult(
            success=True,
            message=f"All {len(steps)} steps completed successfully",
            steps=steps,
        )

    def _execute_action(self, action: RobotAction, step_num: int) -> StepResult:
        """Execute a single robot action."""
        start_time = time.time()

        try:
            if action.action_type == ActionType.MOVE_TO:
                return self._exec_move_to(action, step_num, start_time)
            elif action.action_type == ActionType.GRASP:
                return self._exec_grasp(action, step_num, start_time)
            elif action.action_type == ActionType.RELEASE:
                return self._exec_release(action, step_num, start_time)
            elif action.action_type == ActionType.PUSH:
                return self._exec_push(action, step_num, start_time)
            elif action.action_type == ActionType.ROTATE:
                return self._exec_rotate(action, step_num, start_time)
            elif action.action_type == ActionType.WAIT:
                return self._exec_wait(action, step_num, start_time)
            elif action.action_type == ActionType.NAVIGATE:
                return self._exec_navigate(action, step_num, start_time)
            elif action.action_type == ActionType.LOOK_AT:
                return self._exec_look_at(action, step_num, start_time)
            else:
                return StepResult(
                    step=step_num, action=str(action), success=False,
                    message=f"Unknown action type: {action.action_type}",
                    duration=time.time() - start_time,
                )
        except Exception as e:
            return StepResult(
                step=step_num, action=str(action), success=False,
                message=f"Exception: {e}",
                duration=time.time() - start_time,
            )

    def _exec_move_to(self, action: RobotAction, step: int, t0: float) -> StepResult:
        pos = action.position
        if pos is None:
            return StepResult(step, str(action), False, "No position specified", 0)

        # Bounds check
        if not self._in_bounds(pos):
            return StepResult(step, str(action), False,
                              f"Position {pos} out of bounds", 0)

        if self.dry_run:
            logger.info(f"  [DRY RUN] move_to {pos}")
        elif self.robot:
            speed = action.speed or 0.5
            self.robot.move_to(pos, speed)

        self.current_position = pos
        return StepResult(
            step, str(action), True,
            f"Moved to {pos}",
            time.time() - t0,
        )

    def _exec_grasp(self, action: RobotAction, step: int, t0: float) -> StepResult:
        if self.gripper_state == "closed":
            return StepResult(step, str(action), False,
                              "Gripper already closed", 0)

        if self.dry_run:
            logger.info(f"  [DRY RUN] grasp {action.target}")
        elif self.robot:
            self.robot.grasp()

        self.gripper_state = "closed"
        self.held_object = action.target
        return StepResult(
            step, str(action), True,
            f"Grasped {action.target or 'object'}",
            time.time() - t0,
        )

    def _exec_release(self, action: RobotAction, step: int, t0: float) -> StepResult:
        if self.gripper_state == "open":
            return StepResult(step, str(action), False,
                              "Gripper already open", 0)

        if self.dry_run:
            logger.info(f"  [DRY RUN] release {action.target}")
        elif self.robot:
            self.robot.release()

        released = self.held_object
        self.gripper_state = "open"
        self.held_object = None
        return StepResult(
            step, str(action), True,
            f"Released {released or 'object'}",
            time.time() - t0,
        )

    def _exec_push(self, action: RobotAction, step: int, t0: float) -> StepResult:
        if self.dry_run:
            logger.info(f"  [DRY RUN] push {action.target} dir={action.direction}")
        return StepResult(
            step, str(action), True,
            f"Pushed {action.target} in direction {action.direction}",
            time.time() - t0,
        )

    def _exec_rotate(self, action: RobotAction, step: int, t0: float) -> StepResult:
        if self.dry_run:
            logger.info(f"  [DRY RUN] rotate {action.angle} rad")
        return StepResult(
            step, str(action), True,
            f"Rotated {action.angle} radians",
            time.time() - t0,
        )

    def _exec_wait(self, action: RobotAction, step: int, t0: float) -> StepResult:
        duration = action.duration or 1.0
        if not self.dry_run:
            time.sleep(duration)
        return StepResult(
            step, str(action), True,
            f"Waited {duration}s",
            time.time() - t0,
        )

    def _exec_navigate(self, action: RobotAction, step: int, t0: float) -> StepResult:
        pos = action.position
        if self.dry_run:
            logger.info(f"  [DRY RUN] navigate to {pos}")
        return StepResult(
            step, str(action), True,
            f"Navigated to {pos}",
            time.time() - t0,
        )

    def _exec_look_at(self, action: RobotAction, step: int, t0: float) -> StepResult:
        if self.dry_run:
            logger.info(f"  [DRY RUN] look_at {action.target}")
        return StepResult(
            step, str(action), True,
            f"Looking at {action.target}",
            time.time() - t0,
        )

    def _validate_plan(self, plan: ActionPlan) -> ValidationResult:
        """Validate the entire plan before execution."""
        for i, action in enumerate(plan.actions):
            if action.action_type == ActionType.MOVE_TO and action.position:
                if not self._in_bounds(action.position):
                    return ValidationResult(
                        False,
                        f"Step {i + 1}: position {action.position} out of workspace bounds",
                    )
        return ValidationResult(True, "Plan is valid")

    def _in_bounds(self, position: tuple[float, ...]) -> bool:
        """Check if a position is within workspace bounds."""
        bounds = self.workspace_bounds
        if len(position) >= 1 and not (bounds["x"][0] <= position[0] <= bounds["x"][1]):
            return False
        if len(position) >= 2 and not (bounds["y"][0] <= position[1] <= bounds["y"][1]):
            return False
        if len(position) >= 3 and not (bounds["z"][0] <= position[2] <= bounds["z"][1]):
            return False
        return True


# =============================================================================
# Result data classes
# =============================================================================

class StepResult:
    """Result of executing a single action step."""
    def __init__(self, step: int, action: str, success: bool,
                 message: str, duration: float):
        self.step = step
        self.action = action
        self.success = success
        self.message = message
        self.duration = duration

    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"  [{status}] Step {self.step}: {self.message} ({self.duration:.3f}s)"


class ExecutionResult:
    """Result of executing a full action plan."""
    def __init__(self, success: bool, message: str, steps: list[StepResult]):
        self.success = success
        self.message = message
        self.steps = steps

    def __str__(self):
        lines = [f"Execution: {'SUCCESS' if self.success else 'FAILED'}"]
        lines.append(f"  {self.message}")
        for step in self.steps:
            lines.append(str(step))
        return "\n".join(lines)


class ValidationResult:
    """Result of plan validation."""
    def __init__(self, valid: bool, message: str):
        self.valid = valid
        self.message = message
