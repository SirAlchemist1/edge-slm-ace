"""Core ACE logic: runner and role-based prompts."""

from edge_slm_ace.core.runner import (
    run_dataset_baseline,
    run_dataset_ace,
    run_dataset_self_refine,
)
from edge_slm_ace.core.ace_roles import (
    build_generator_prompt,
    build_reflector_prompt,
    parse_generator_output,
    parse_reflector_output_to_lessons,
    choose_lessons_for_playbook,
)

__all__ = [
    "run_dataset_baseline",
    "run_dataset_ace",
    "run_dataset_self_refine",
    "build_generator_prompt",
    "build_reflector_prompt",
    "parse_generator_output",
    "parse_reflector_output_to_lessons",
    "choose_lessons_for_playbook",
]
