"""
Prompt templates for the LLM-based robot planner.
These templates structure the LLM's input so it produces well-formed
action sequences grounded in the perceived environment.
"""

# =============================================================================
# System Prompt — Defines the LLM's role and output format
# =============================================================================

SYSTEM_PROMPT = """\
You are an intelligent robot control planner. Your job is to translate high-level
human instructions into precise sequences of robot primitive actions.

You have access to the following action primitives:

| Action       | Parameters                                | Description                          |
|-------------|-------------------------------------------|--------------------------------------|
| move_to     | target, position (x, y, z)                | Move end-effector to a position      |
| grasp       | target                                    | Close gripper to grasp an object     |
| release     | target                                    | Open gripper to release an object    |
| push        | target, direction (dx, dy, dz)            | Push an object in a direction        |
| rotate      | angle (radians)                           | Rotate end-effector                  |
| wait        | duration (seconds)                        | Pause execution                      |
| navigate    | target, position (x, y)                   | Move mobile base to a location       |
| look_at     | target, position (x, y, z)               | Orient camera toward a target        |

## Rules:
1. You MUST output actions as a valid JSON array.
2. Each action must have: "action", "params", and "description" fields.
3. Always consider the current positions of objects from the scene description.
4. Think step-by-step before producing the action sequence.
5. If the task is impossible given the current scene, explain why and return an empty action list.
6. Consider safety: avoid collisions, ensure grasps before moving objects, etc.
7. Be precise with coordinates — use the positions provided in the scene description.

## Output Format:
```json
{
  "reasoning": "Step-by-step explanation of the plan...",
  "actions": [
    {
      "action": "move_to",
      "params": {"target": "object_name", "position": [x, y, z]},
      "description": "Human-readable description of this step"
    }
  ],
  "feasible": true,
  "error": null
}
```
"""


# =============================================================================
# User Prompt Template — Combines task + scene context
# =============================================================================

USER_PROMPT_TEMPLATE = """\
## Task
{task}

{scene_description}

## Robot State
- Current end-effector position: {robot_position}
- Gripper state: {gripper_state}

Please generate the action sequence to accomplish this task. Output valid JSON only.
"""


# =============================================================================
# Few-Shot Examples (optional, improves output quality)
# =============================================================================

FEW_SHOT_EXAMPLES = [
    {
        "task": "Pick up the red block and place it on the blue platform.",
        "scene": (
            "Objects: red block at (0.3, 0.5, 0.02), blue platform at (0.7, 0.5, 0.0), "
            "table surface at (0.5, 0.5, 0.0). "
            "Relations: red block is on table, blue platform is on table."
        ),
        "output": {
            "reasoning": (
                "1. The red block is at (0.3, 0.5, 0.02) on the table. "
                "2. I need to move above it, descend, grasp it, lift it, "
                "move to the blue platform at (0.7, 0.5, 0.0), descend, "
                "and release."
            ),
            "actions": [
                {
                    "action": "move_to",
                    "params": {"target": "red block", "position": [0.3, 0.5, 0.15]},
                    "description": "Move above the red block"
                },
                {
                    "action": "move_to",
                    "params": {"target": "red block", "position": [0.3, 0.5, 0.02]},
                    "description": "Descend to the red block"
                },
                {
                    "action": "grasp",
                    "params": {"target": "red block"},
                    "description": "Grasp the red block"
                },
                {
                    "action": "move_to",
                    "params": {"target": "red block", "position": [0.3, 0.5, 0.2]},
                    "description": "Lift the red block"
                },
                {
                    "action": "move_to",
                    "params": {"target": "blue platform", "position": [0.7, 0.5, 0.15]},
                    "description": "Move above the blue platform"
                },
                {
                    "action": "move_to",
                    "params": {"target": "blue platform", "position": [0.7, 0.5, 0.02]},
                    "description": "Descend to the blue platform"
                },
                {
                    "action": "release",
                    "params": {"target": "red block"},
                    "description": "Release the red block on the blue platform"
                },
            ],
            "feasible": True,
            "error": None,
        },
    },
]


def build_few_shot_messages() -> list[dict]:
    """Build few-shot example messages for the LLM."""
    import json

    messages = []
    for ex in FEW_SHOT_EXAMPLES:
        # User turn
        messages.append({
            "role": "user",
            "content": (
                f"## Task\n{ex['task']}\n\n"
                f"## Scene\n{ex['scene']}\n\n"
                "## Robot State\n"
                "- Current end-effector position: (0.5, 0.5, 0.3)\n"
                "- Gripper state: open\n\n"
                "Please generate the action sequence. Output valid JSON only."
            ),
        })
        # Assistant turn
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["output"], indent=2),
        })

    return messages
