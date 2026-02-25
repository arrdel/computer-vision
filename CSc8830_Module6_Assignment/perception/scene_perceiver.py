"""
Scene Perception Module
-----------------------
Uses vision-language models (CLIP / OpenCLIP) to understand the environment
from camera images. Produces a SceneGraph with detected objects and spatial
relationships that can be fed to the LLM planner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from models.data_models import (
    BoundingBox,
    SceneGraph,
    SceneObject,
    SpatialRelation,
)

logger = logging.getLogger(__name__)


# Default vocabulary of common objects the robot might interact with
DEFAULT_OBJECT_VOCAB = [
    "red block", "blue block", "green block", "yellow block",
    "red cup", "blue cup", "green cup",
    "plate", "bowl", "bottle", "can",
    "table", "shelf", "platform", "tray",
    "ball", "box", "cylinder", "cone",
    "chair", "door", "drawer", "button",
    "apple", "banana", "orange",
    "pen", "book", "phone", "remote",
    "robot arm", "gripper", "nothing / empty space",
]


class ScenePerceiver:
    """
    Perceives and understands a scene from an image using CLIP.

    Uses CLIP to:
    1. Identify objects in the scene by matching against a vocabulary.
    2. Determine spatial relationships between objects.
    3. Build a SceneGraph for downstream planning.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        object_vocab: Optional[list[str]] = None,
        confidence_threshold: float = 0.25,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.confidence_threshold = confidence_threshold
        self.object_vocab = object_vocab or DEFAULT_OBJECT_VOCAB

        # Determine device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self) -> None:
        """Lazy-load the CLIP model."""
        if self._loaded:
            return

        try:
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model = self.model.to(self.device).eval()
            self._loaded = True
            logger.info(
                f"Loaded CLIP model {self.model_name} on {self.device}"
            )
        except ImportError:
            logger.warning(
                "open_clip not installed. Using mock perception. "
                "Install with: pip install open-clip-torch"
            )
            self._loaded = False

    def perceive(
        self,
        image_source: str | Path | Image.Image | np.ndarray,
        custom_vocab: Optional[list[str]] = None,
    ) -> SceneGraph:
        """
        Perceive the scene from an image.

        Args:
            image_source: Path to image, PIL Image, or numpy array.
            custom_vocab: Optional custom object vocabulary to search for.

        Returns:
            SceneGraph with detected objects and spatial relations.
        """
        self.load_model()

        # Load image
        image = self._load_image(image_source)
        vocab = custom_vocab or self.object_vocab

        if not self._loaded:
            logger.warning("Model not loaded — returning mock scene")
            return self._mock_scene(image_source)

        # Encode image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Encode text prompts
        text_prompts = [f"a photo of a {obj}" for obj in vocab]
        text_tokens = self.tokenizer(text_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            similarities = (image_features @ text_features.T).squeeze(0)
            probs = similarities.softmax(dim=-1).cpu().numpy()

        # Extract detected objects
        objects = []
        for idx, (obj_name, prob) in enumerate(zip(vocab, probs)):
            if prob >= self.confidence_threshold:
                # Assign approximate positions based on detection order
                # (In a real system, you'd use a detector for bounding boxes)
                position = self._estimate_position(idx, len(vocab), image.size)
                category = self._infer_category(obj_name)

                objects.append(
                    SceneObject(
                        name=obj_name,
                        category=category,
                        position=position,
                        color=self._extract_color(obj_name),
                        confidence=float(prob),
                    )
                )

        # Sort by confidence
        objects.sort(key=lambda o: o.confidence, reverse=True)

        # Infer spatial relations
        relations = self._infer_spatial_relations(objects)

        # Build description
        scene = SceneGraph(
            objects=objects,
            relations=relations,
            image_path=str(image_source) if isinstance(image_source, (str, Path)) else None,
        )
        scene.description = scene.to_text()

        logger.info(f"Perceived {len(objects)} objects with {len(relations)} relations")
        return scene

    def perceive_with_description(
        self,
        image_source: str | Path | Image.Image,
        vlm_description: Optional[str] = None,
    ) -> SceneGraph:
        """
        Perceive scene using CLIP + an optional VLM-generated description.
        This allows integrating richer descriptions from models like GPT-4V.
        """
        scene = self.perceive(image_source)

        if vlm_description:
            scene.description = (
                vlm_description + "\n\n" + scene.to_text()
            )
        return scene

    # ---- Private helpers ----

    def _load_image(
        self, source: str | Path | Image.Image | np.ndarray
    ) -> Image.Image:
        """Load an image from various sources."""
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")
        return Image.open(source).convert("RGB")

    def _estimate_position(
        self, idx: int, total: int, image_size: tuple[int, int]
    ) -> tuple[float, float]:
        """
        Rough position estimate. In production, use an object detector
        (e.g., OWL-ViT, Grounding DINO) for accurate bounding boxes.
        """
        w, h = image_size
        # Spread objects across a normalized workspace
        cols = max(int(total**0.5), 1)
        row = idx // cols
        col = idx % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / ((total // cols) + 1)
        return (round(x, 3), round(y, 3))

    def _infer_category(self, name: str) -> str:
        """Infer a coarse category from object name."""
        categories = {
            "block": "manipulable",
            "cup": "container",
            "plate": "surface",
            "bowl": "container",
            "bottle": "container",
            "can": "container",
            "table": "furniture",
            "shelf": "furniture",
            "platform": "surface",
            "tray": "surface",
            "ball": "manipulable",
            "box": "container",
            "chair": "furniture",
            "door": "fixture",
            "drawer": "fixture",
            "button": "interactable",
            "apple": "manipulable",
            "banana": "manipulable",
            "orange": "manipulable",
            "pen": "manipulable",
            "book": "manipulable",
            "phone": "manipulable",
            "arm": "robot",
            "gripper": "robot",
        }
        name_lower = name.lower()
        for keyword, cat in categories.items():
            if keyword in name_lower:
                return cat
        return "object"

    def _extract_color(self, name: str) -> Optional[str]:
        """Extract color from object name if present."""
        colors = ["red", "blue", "green", "yellow", "orange", "purple",
                  "white", "black", "brown", "pink", "gray", "grey"]
        name_lower = name.lower()
        for color in colors:
            if color in name_lower:
                return color
        return None

    def _infer_spatial_relations(
        self, objects: list[SceneObject]
    ) -> list[SpatialRelation]:
        """Infer spatial relationships between detected objects."""
        relations = []
        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i == j:
                    continue
                rel = self._compute_relation(obj_a, obj_b)
                if rel:
                    relations.append(rel)
        return relations

    def _compute_relation(
        self, a: SceneObject, b: SceneObject
    ) -> Optional[SpatialRelation]:
        """Compute the dominant spatial relation between two objects."""
        ax, ay = a.position[0], a.position[1]
        bx, by = b.position[0], b.position[1]

        dx = ax - bx
        dy = ay - by
        dist = (dx**2 + dy**2) ** 0.5

        # Only report relations for nearby objects
        if dist > 0.5:
            return None

        if dist < 0.1:
            return SpatialRelation(a.name, "near", b.name, confidence=0.9)

        if abs(dx) > abs(dy):
            rel = "right_of" if dx > 0 else "left_of"
        else:
            rel = "below" if dy > 0 else "above"

        return SpatialRelation(a.name, rel, b.name, confidence=0.7)

    def _mock_scene(
        self, image_source: str | Path | Image.Image | np.ndarray = None,
    ) -> SceneGraph:
        """Return a mock scene for testing without a loaded model."""
        objects = [
            SceneObject("red block", "manipulable", (0.3, 0.5), color="red",
                        size="small", confidence=0.95),
            SceneObject("blue block", "manipulable", (0.5, 0.5), color="blue",
                        size="small", confidence=0.90),
            SceneObject("table", "furniture", (0.5, 0.8), size="large",
                        confidence=0.98),
            SceneObject("green cup", "container", (0.7, 0.4), color="green",
                        confidence=0.85),
        ]
        relations = [
            SpatialRelation("red block", "on", "table"),
            SpatialRelation("blue block", "on", "table"),
            SpatialRelation("green cup", "on", "table"),
            SpatialRelation("red block", "left_of", "blue block"),
            SpatialRelation("blue block", "left_of", "green cup"),
        ]
        scene = SceneGraph(objects=objects, relations=relations)
        scene.description = scene.to_text()
        return scene
