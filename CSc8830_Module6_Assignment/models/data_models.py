"""Data models for scene representation and robot actions."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =============================================================================
# Scene Representation
# =============================================================================

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in 3D (or 2D if z is None)."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    z_min: Optional[float] = None
    z_max: Optional[float] = None

    @property
    def center(self) -> tuple[float, ...]:
        cx = (self.x_min + self.x_max) / 2
        cy = (self.y_min + self.y_max) / 2
        if self.z_min is not None and self.z_max is not None:
            cz = (self.z_min + self.z_max) / 2
            return (cx, cy, cz)
        return (cx, cy)


@dataclass
class SceneObject:
    """A single object detected in the scene."""
    name: str
    category: str
    position: tuple[float, ...]        # (x, y) or (x, y, z)
    bbox: Optional[BoundingBox] = None
    color: Optional[str] = None
    size: Optional[str] = None          # "small", "medium", "large"
    state: Optional[str] = None         # "open", "closed", "stacked", etc.
    confidence: float = 1.0
    properties: dict = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [self.name]
        if self.color:
            parts.insert(0, self.color)
        if self.size:
            parts.insert(0, self.size)
        return " ".join(parts)


@dataclass
class SpatialRelation:
    """A spatial relationship between two objects."""
    subject: str        # Object name
    relation: str       # e.g., "on", "left_of", "near", "inside"
    reference: str      # Reference object name
    confidence: float = 1.0


@dataclass
class SceneGraph:
    """Complete scene representation with objects and their spatial relations."""
    objects: list[SceneObject] = field(default_factory=list)
    relations: list[SpatialRelation] = field(default_factory=list)
    description: str = ""
    image_path: Optional[str] = None
    timestamp: Optional[float] = None

    def to_text(self) -> str:
        """Convert scene graph to a natural language description for the LLM."""
        lines = ["## Current Scene Description", ""]

        # Objects
        lines.append("### Objects in the environment:")
        for i, obj in enumerate(self.objects, 1):
            pos_str = ", ".join(f"{v:.2f}" for v in obj.position)
            desc = f"{i}. **{obj}** (category: {obj.category}) at position ({pos_str})"
            if obj.state:
                desc += f" — state: {obj.state}"
            if obj.confidence < 1.0:
                desc += f" [confidence: {obj.confidence:.0%}]"
            lines.append(desc)

        # Spatial relations
        if self.relations:
            lines.append("")
            lines.append("### Spatial relationships:")
            for rel in self.relations:
                lines.append(
                    f"- {rel.subject} is **{rel.relation}** {rel.reference}"
                )

        return "\n".join(lines)

    def get_object_by_name(self, name: str) -> Optional[SceneObject]:
        """Find an object by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for obj in self.objects:
            if name_lower in obj.name.lower() or name_lower in str(obj).lower():
                return obj
        return None


# =============================================================================
# Robot Actions
# =============================================================================

class ActionType(Enum):
    """Primitive robot action types."""
    MOVE_TO = "move_to"
    GRASP = "grasp"
    RELEASE = "release"
    PUSH = "push"
    ROTATE = "rotate"
    WAIT = "wait"
    NAVIGATE = "navigate"
    LOOK_AT = "look_at"


@dataclass
class RobotAction:
    """A single robot action with parameters."""
    action_type: ActionType
    target: Optional[str] = None                 # Target object name
    position: Optional[tuple[float, ...]] = None # Target position
    direction: Optional[tuple[float, ...]] = None
    angle: Optional[float] = None                # Rotation angle (radians)
    duration: Optional[float] = None             # Duration (seconds)
    speed: Optional[float] = None                # Speed multiplier 0-1
    description: str = ""                        # Human-readable description

    def __str__(self) -> str:
        parts = [f"[{self.action_type.value}]"]
        if self.target:
            parts.append(f"target={self.target}")
        if self.position:
            pos_str = ", ".join(f"{v:.3f}" for v in self.position)
            parts.append(f"pos=({pos_str})")
        if self.description:
            parts.append(f'"{self.description}"')
        return " ".join(parts)


@dataclass
class ActionPlan:
    """An ordered sequence of robot actions produced by the planner."""
    task: str                                      # Original high-level task
    actions: list[RobotAction] = field(default_factory=list)
    scene_context: Optional[SceneGraph] = None
    reasoning: str = ""                            # LLM's chain-of-thought reasoning
    success: bool = True
    error_message: str = ""

    def __str__(self) -> str:
        lines = [f"Plan for: '{self.task}'", f"Steps ({len(self.actions)}):"]
        for i, action in enumerate(self.actions, 1):
            lines.append(f"  {i}. {action}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)
