"""
LLM-Based Robot Action Planner
-------------------------------
Takes a high-level natural language task instruction and a SceneGraph,
then queries an LLM to produce a structured ActionPlan of robot primitives.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from models.data_models import (
    ActionPlan,
    ActionType,
    RobotAction,
    SceneGraph,
    SceneObject,
)
from prompts.planner_prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_few_shot_messages,
)

logger = logging.getLogger(__name__)


class LLMPlanner:
    """
    Plans robot actions by prompting an LLM with scene context.

    The planner:
    1. Formats the scene graph + task into a structured prompt.
    2. Sends it to an LLM (OpenAI, local, etc.).
    3. Parses the JSON response into an ActionPlan.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        use_few_shot: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_few_shot = use_few_shot

        # Resolve API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

    def _init_client(self):
        """Initialize the LLM client lazily."""
        if self.client is not None:
            return

        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        elif self.provider == "local":
            # For local models (e.g., via Ollama or vLLM), point to local server
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="http://localhost:11434/v1",  # Ollama default
                    api_key="ollama",
                )
            except ImportError:
                raise ImportError("openai package required for local LLM client.")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def plan(
        self,
        task: str,
        scene: SceneGraph,
        robot_position: tuple[float, ...] = (0.5, 0.5, 0.3),
        gripper_state: str = "open",
    ) -> ActionPlan:
        """
        Generate an action plan for the given task in the given scene.

        Args:
            task: High-level natural language instruction.
            scene: Current scene understanding (SceneGraph).
            robot_position: Current end-effector position.
            gripper_state: Current gripper state ("open" or "closed").

        Returns:
            ActionPlan with ordered list of RobotActions.
        """
        self._init_client()

        # Build the prompt
        messages = self._build_messages(
            task, scene, robot_position, gripper_state
        )

        logger.info(f"Planning for task: '{task}' with {len(scene.objects)} objects")
        logger.debug(f"Prompt messages: {len(messages)} messages")

        # Query the LLM
        try:
            response = self._query_llm(messages)
            plan = self._parse_response(response, task, scene)
            logger.info(f"Generated plan with {len(plan)} actions")
            return plan
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return ActionPlan(
                task=task,
                scene_context=scene,
                success=False,
                error_message=str(e),
            )

    def plan_mock(
        self,
        task: str,
        scene: SceneGraph,
    ) -> ActionPlan:
        """
        Generate a mock plan without calling an LLM.
        Useful for testing the pipeline end-to-end.
        """
        logger.info(f"[MOCK] Planning for: '{task}'")

        # Simple heuristic planning for common tasks
        actions = []
        task_lower = task.lower()

        if "pick" in task_lower and "place" in task_lower:
            # Extract object references from the task
            pick_obj = self._find_referenced_object(task, scene, hint="pick")
            place_obj = self._find_referenced_object(task, scene, hint="place")

            if pick_obj:
                px, py = pick_obj.position[0], pick_obj.position[1]
                actions.extend([
                    RobotAction(ActionType.MOVE_TO, pick_obj.name,
                                position=(px, py, 0.15),
                                description=f"Move above {pick_obj.name}"),
                    RobotAction(ActionType.MOVE_TO, pick_obj.name,
                                position=(px, py, 0.02),
                                description=f"Descend to {pick_obj.name}"),
                    RobotAction(ActionType.GRASP, pick_obj.name,
                                description=f"Grasp {pick_obj.name}"),
                    RobotAction(ActionType.MOVE_TO, pick_obj.name,
                                position=(px, py, 0.2),
                                description=f"Lift {pick_obj.name}"),
                ])

            if place_obj:
                tx, ty = place_obj.position[0], place_obj.position[1]
                obj_name = pick_obj.name if pick_obj else "object"
                actions.extend([
                    RobotAction(ActionType.MOVE_TO, place_obj.name,
                                position=(tx, ty, 0.15),
                                description=f"Move above {place_obj.name}"),
                    RobotAction(ActionType.MOVE_TO, place_obj.name,
                                position=(tx, ty, 0.05),
                                description=f"Descend to {place_obj.name}"),
                    RobotAction(ActionType.RELEASE, obj_name,
                                description=f"Release {obj_name} on {place_obj.name}"),
                ])

        elif "push" in task_lower:
            obj = self._find_referenced_object(task, scene)
            if obj:
                actions.append(
                    RobotAction(ActionType.PUSH, obj.name,
                                direction=(0.1, 0.0, 0.0),
                                description=f"Push {obj.name} forward")
                )
        else:
            # Default: try to move to the first mentioned object
            obj = self._find_referenced_object(task, scene)
            if obj:
                px, py = obj.position[0], obj.position[1]
                actions.append(
                    RobotAction(ActionType.MOVE_TO, obj.name,
                                position=(px, py, 0.1),
                                description=f"Move to {obj.name}")
                )

        return ActionPlan(
            task=task,
            actions=actions,
            scene_context=scene,
            reasoning="[MOCK] Heuristic plan generated without LLM.",
        )

    # ---- Private methods ----

    def _build_messages(
        self,
        task: str,
        scene: SceneGraph,
        robot_position: tuple,
        gripper_state: str,
    ) -> list[dict]:
        """Construct the message list for the LLM API call."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Optionally include few-shot examples
        if self.use_few_shot:
            messages.extend(build_few_shot_messages())

        # Build the user prompt
        pos_str = ", ".join(f"{v:.2f}" for v in robot_position)
        user_content = USER_PROMPT_TEMPLATE.format(
            task=task,
            scene_description=scene.to_text(),
            robot_position=f"({pos_str})",
            gripper_state=gripper_state,
        )
        messages.append({"role": "user", "content": user_content})

        return messages

    def _query_llm(self, messages: list[dict]) -> str:
        """Send messages to the LLM and return the response text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _parse_response(
        self, response_text: str, task: str, scene: SceneGraph
    ) -> ActionPlan:
        """Parse the LLM's JSON response into an ActionPlan."""
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r"```json?\s*(.*?)```", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                raise ValueError(f"Failed to parse LLM response as JSON: {e}")

        # Check feasibility
        if not data.get("feasible", True):
            return ActionPlan(
                task=task,
                scene_context=scene,
                reasoning=data.get("reasoning", ""),
                success=False,
                error_message=data.get("error", "Task deemed infeasible by planner"),
            )

        # Parse actions
        actions = []
        for action_data in data.get("actions", []):
            action_type = ActionType(action_data["action"])
            params = action_data.get("params", {})

            action = RobotAction(
                action_type=action_type,
                target=params.get("target"),
                position=tuple(params["position"]) if "position" in params else None,
                direction=tuple(params["direction"]) if "direction" in params else None,
                angle=params.get("angle"),
                duration=params.get("duration"),
                speed=params.get("speed"),
                description=action_data.get("description", ""),
            )
            actions.append(action)

        return ActionPlan(
            task=task,
            actions=actions,
            scene_context=scene,
            reasoning=data.get("reasoning", ""),
            success=True,
        )

    def _find_referenced_object(
        self, task: str, scene: SceneGraph, hint: str = ""
    ) -> Optional[SceneObject]:
        """Find the object in the scene most likely referenced by the task."""
        task_lower = task.lower()
        best_match = None
        best_score = 0

        for obj in scene.objects:
            score = 0
            obj_words = obj.name.lower().split()
            for word in obj_words:
                if word in task_lower:
                    score += 1

            # Boost score if near the hint keyword
            if hint:
                hint_pos = task_lower.find(hint)
                if hint_pos >= 0:
                    obj_pos = task_lower.find(obj.name.lower())
                    if obj_pos >= 0:
                        distance = abs(obj_pos - hint_pos)
                        score += max(0, 5 - distance / 10)

            if score > best_score:
                best_score = score
                best_match = obj

        return best_match
