# LLM-Driven Robot Control with Environmental Understanding

## Overview

This project implements a system that uses a Large Language Model (LLM) to generate
robot control sequences from high-level natural language prompts. The system perceives
the environment through vision (camera/images), builds a semantic scene description,
and uses an LLM to plan and produce executable robot actions.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌──────────────┐
│  Camera /    │────▶│  Scene Perception │────▶│  LLM Planner │────▶│  Action      │
│  Image Input │     │  (CLIP / VLM)     │     │  (GPT/Llama) │     │  Executor    │
└─────────────┘     └──────────────────┘     └─────────────┘     └──────────────┘
                           │                        │                     │
                           ▼                        ▼                     ▼
                    Scene Description          Action Plan          Robot Commands
                    (objects, spatial          (structured          (velocities,
                     relations, state)         action sequence)     waypoints, gripper)
```

## Modules

- **`perception/`** — Environment understanding via vision-language models
- **`planner/`** — LLM-based action planning from prompts + scene context
- **`executor/`** — Translates action plans into robot control commands
- **`simulation/`** — Simple 2D simulation environment for testing
- **`prompts/`** — Prompt templates for the LLM planner
- **`configs/`** — Configuration files

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key (if using OpenAI)
export OPENAI_API_KEY="your-key-here"
```

## Usage

```bash
# Run with a simulated environment
python main.py --mode sim --task "pick up the red block and place it on the blue platform"

# Run with an image input
python main.py --mode image --image path/to/scene.jpg --task "navigate to the chair"

# Run the interactive demo
python demo.py
```

## Example

```python
from perception.scene_perceiver import ScenePerceiver
from planner.llm_planner import LLMPlanner
from executor.action_executor import ActionExecutor

# Perceive the environment
perceiver = ScenePerceiver()
scene = perceiver.perceive("scene.jpg")

# Plan actions from a high-level prompt
planner = LLMPlanner()
plan = planner.plan("pick up the red cup and place it on the table", scene)

# Execute the plan
executor = ActionExecutor()
executor.execute(plan)
```
